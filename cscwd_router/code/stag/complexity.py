"""CPU operator counts for the native, fixed-budget DINOv3 Router graph."""

from collections import defaultdict
import hashlib
import importlib
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from gazelle.routing.backbone import RoutedDinoV3Backbone
from gazelle.routing.model import CoverageAwareGazeLLE


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_source_hash(repo):
    digest = hashlib.sha256()
    for path in sorted(Path(repo).rglob("*.py")):
        name = str(Path("dinov3") / path.relative_to(repo))
        digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def attention_flops(query, key, value, *args, **kwargs):
    """QK^T and AV, with one multiplication and one addition per MAC."""
    return 2 * math.prod(query[:-2]) * query[-2] * key[-2] * (query[-1] + value[-1])


def flop_counter():
    # PyTorch's default registry omits the fused CPU SDPA operator. Without
    # this entry the attention matrix products silently disappear from MACs.
    return FlopCounterMode(display=False, custom_mapping={
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: attention_flops,
    })


class ArchitectureBackbone(RoutedDinoV3Backbone):
    """Native architecture without reading or downloading pretrained weights."""

    def __init__(self, size, native_repo):
        nn.Module.__init__(self)
        sys.path.insert(0, str(Path(native_repo).resolve()))
        factories = importlib.import_module("dinov3.hub.backbones")
        self.model = getattr(factories, f"dinov3_vit{size}16")(pretrained=False)
        self.out_indices = [2, 5, 8, 11] if size == "b" else [5, 11, 17, 23]


def build_architecture(summary, native_repo):
    config = dict(summary["model_config"])
    name = config.pop("model")
    model = CoverageAwareGazeLLE(ArchitectureBackbone(summary["size"], native_repo),
                                inout=name.endswith("_inout"), **config)
    model.model_name = name
    if model.get_model_config() != summary["model_config"]:
        raise ValueError("profile architecture differs from the selected model")
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def profile(model, inputs):
    trace = []
    hooks = [block.register_forward_pre_hook(
        lambda _module, values, index=index: trace.append({"block": index,
            "batch": values[0].shape[0], "tokens": values[0].shape[1]})
    ) for index, block in enumerate(model.backbone.model.blocks)]
    try:
        with flop_counter() as counter:
            output = model(inputs)
    finally:
        for hook in hooks:
            hook.remove()

    counts = counter.get_flop_counts()
    global_ops = {str(op): int(value) for op, value in counts["Global"].items()}
    attention = sum(value for op, value in global_ops.items() if "attention" in op or "bmm" in op)
    if attention == 0:
        raise RuntimeError("attention matrix products were not counted")
    label = type(model).__name__
    subtotal = lambda name: sum(counts.get(f"{label}.{name}", {}).values())
    stages = defaultdict(int)
    stages["token_embedding_and_prefix"] = subtotal("backbone.model.patch_embed")
    for index in range(len(model.backbone.model.blocks)):
        name = "token_embedding_and_prefix" if index <= model.route_after_block else "sparse_suffix"
        stages[name] += subtotal(f"backbone.model.blocks.{index}")
    stages["router"] = subtotal("router")
    total = counter.get_total_flops()
    stages["multilayer_decoder_and_outputs"] = total - sum(stages.values())
    batch = inputs["images"].shape[0]
    kept = output["routing"].image_keep_indices.shape[1]
    expected = [{"block": index, "batch": batch,
        "tokens": (1024 if index <= model.route_after_block else kept) + 5}
        for index in range(len(model.backbone.model.blocks))]
    if trace != expected:
        raise ValueError("profile token graph differs from the trained architecture")
    return {"macs_per_image": total // (2 * batch), "flops_per_image": total // batch,
        "stage_macs_per_image": {name: value // (2 * batch) for name, value in stages.items()},
        "counted_operator_flops_per_batch": global_ops, "attention_matmul_flops_per_batch": attention,
        "block_inputs": trace, "kept_patches": kept,
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
            "task_modules": sum(p.numel() for name, p in model.named_parameters() if not name.startswith("backbone."))}}


def summarize(cells):
    rows = []
    for cell in cells:
        full, sparse = cell["results"]["k100"], cell["results"]["k50"]
        rows.append({"dataset": cell["dataset"], "size": cell["size"],
            "k100_gmacs": full["macs_per_image"] / 1e9,
            "k50_gmacs": sparse["macs_per_image"] / 1e9,
            "mac_reduction_fraction": 1 - sparse["macs_per_image"] / full["macs_per_image"],
            "stage_macs_saved": {name: full["stage_macs_per_image"][name] - value
                for name, value in sparse["stage_macs_per_image"].items()}})
    return rows
