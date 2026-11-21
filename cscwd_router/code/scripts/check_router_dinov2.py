"""T2 server smoke: fit synthetic support using frozen official DINOv2 features.

This checks optimization and routing, not GazeFollow accuracy or efficiency.
Run manually on the server from the repository root. No trained weights are saved.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dinov2"))

from gazelle.routing.losses import coverage_router_loss
from cscwd_router.code.data_prep.synthetic_support import make_synthetic_support_batch
from cscwd_router.code.scripts.check_sparse_dinov2 import compare, sha256
from cscwd_router.code.stag.backbone_sparse_dinov2 import RoutedDinoV2Backbone
from cscwd_router.code.stag.router_dinov2 import GazeConditionedDinoV2Encoder


def fit_support(encoder, features, bboxes, targets, *, steps, learning_rate, budget_weight):
    """Server-only optimization on one cached frozen prefix batch."""
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.router.parameters(), lr=learning_rate, weight_decay=0.0)

    def losses():
        routing = encoder.route_features(features, bboxes)
        return coverage_router_loss(routing, targets, budget_weight=budget_weight, entropy_weight=0.0)

    def snapshot(step):
        with torch.no_grad():
            return {"step": step, **{name: value.item() for name, value in losses().items()}}

    history = [snapshot(0)]
    gradients = {}
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        terms = losses()
        if not torch.isfinite(terms["total"]):
            raise FloatingPointError(f"non-finite synthetic support loss at step {step}")
        terms["total"].backward()
        if step == 2:
            for name in ("scene_projection", "head_projection", "geometry_projection", "score_head"):
                gradients[name] = sum(parameter.grad.abs().sum().item()
                                      for parameter in getattr(encoder.router, name).parameters())
        optimizer.step()
        history.append(snapshot(step))
        if step == 1 or step % 20 == 0 or step == steps:
            print(json.dumps(history[-1]), flush=True)
    return history, gradients


@torch.no_grad()
def check_routes(encoder, images, bboxes, early_features):
    """Check fixed budgets and one suffix per image, including K100 control."""
    encoder.eval()
    batch_size, _, height, width = early_features.shape
    num_patches = height * width
    original_ratio = encoder.router.keep_ratio
    expected_mapping = [image for image, boxes in enumerate(bboxes) for _ in boxes]
    cases = []
    for keep_ratio in (0.25, 0.5, 1.0):
        encoder.router.keep_ratio = keep_ratio
        trace = []
        hooks = [block.register_forward_pre_hook(
            lambda _module, args, index=index: trace.append({"block": index, "batch": args[0].shape[0], "tokens": args[0].shape[1]})
        ) for index, block in enumerate(encoder.backbone.model.blocks)]
        output = encoder(images, bboxes)
        for hook in hooks:
            hook.remove()
        routing = output.routing
        keep_count = round(keep_ratio * num_patches)
        expected_trace = [
            {"block": index, "batch": batch_size,
             "tokens": (num_patches if index <= encoder.route_after_block else keep_count) + encoder.backbone.prefix_token_count}
            for index in range(len(encoder.backbone.model.blocks))
        ]
        union_matches = all(torch.equal(
            routing.image_union_probs[image], routing.support_probs[routing.person_to_image == image].amax(0)
        ) for image in range(batch_size))
        case = {
            "keep_ratio": keep_ratio, "keep_count": keep_count,
            "feature_shape": list(output.features.shape),
            "finite": bool(torch.isfinite(output.features).all() and torch.isfinite(routing.support_probs).all()),
            "person_to_image": routing.person_to_image.tolist(),
            "person_mapping_matches": routing.person_to_image.tolist() == expected_mapping,
            "max_union_matches": union_matches,
            "fixed_budget": routing.image_hard_masks.flatten(1).sum(1).tolist() == [keep_count] * batch_size,
            "sorted_indices": torch.equal(routing.image_keep_indices, routing.image_keep_indices.sort(1).values),
            "normalized_support": torch.allclose(routing.support_probs.flatten(1).sum(1), torch.ones(len(expected_mapping), device=images.device)),
            "block_inputs": trace, "one_shared_image_suffix": trace == expected_trace,
        }
        if keep_ratio < 1.0:
            dropped = ~routing.image_hard_masks
            case["early_exit_fill"] = compare(output.features.permute(0, 2, 3, 1)[dropped],
                                               early_features.permute(0, 2, 3, 1)[dropped])
        else:
            expected = encoder.backbone.model.forward_features(images)["x_norm_patchtokens"]
            case["k100_backbone"] = compare(output.features.flatten(2).transpose(1, 2), expected)
        numerical = case.get("early_exit_fill", case.get("k100_backbone"))
        case["passed"] = (all(case[name] for name in ("finite", "person_mapping_matches", "max_union_matches", "fixed_budget",
                                                      "sorted_indices", "normalized_support", "one_shared_image_suffix"))
                          and numerical["equal"] and numerical["finite"])
        cases.append(case)
    encoder.router.keep_ratio = original_ratio
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, choices=(448,), default=448)
    parser.add_argument("--route-after-block", type=int, choices=(5,), default=5)
    parser.add_argument("--keep-ratio", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--escape-tokens", type=int, default=8)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--budget-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"run from the repository root: {REPO_ROOT}")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in (
        "cscwd_router.code.stag.test_backbone_sparse_dinov2", "cscwd_router.code.stag.test_router_dinov2",
    ))
    synthetic = unittest.TextTestRunner(verbosity=2).run(suite)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    batch = make_synthetic_support_batch(args.batch_size, args.image_size, seed=args.seed)
    images, targets = batch.images.to(device), batch.target_heatmaps.to(device)
    encoder = GazeConditionedDinoV2Encoder(
        RoutedDinoV2Backbone(), route_after_block=args.route_after_block, keep_ratio=args.keep_ratio,
        hidden_dim=args.hidden_dim, temperature=args.temperature, escape_tokens=args.escape_tokens,
    ).to(device)
    _, features = encoder.forward_prefix(images)
    history, gradients = fit_support(encoder, features, batch.bboxes, targets, steps=args.steps,
                                     learning_rate=args.learning_rate, budget_weight=args.budget_weight)
    cases = check_routes(encoder, images, batch.bboxes, features)
    backbone_trainable = sum(parameter.numel() for parameter in encoder.backbone.parameters() if parameter.requires_grad)
    backbone_gradients = sum(parameter.grad is not None for parameter in encoder.backbone.parameters())
    coverage_decreased = history[-1]["coverage"] < history[0]["coverage"]
    branches_connected = bool(gradients) and all(value > 0 for value in gradients.values())
    source_names = [
        "gazelle/routing/router.py", "gazelle/routing/losses.py",
        "cscwd_router/code/stag/backbone_sparse_dinov2.py",
        "cscwd_router/code/stag/router_dinov2.py", "cscwd_router/code/stag/test_router_dinov2.py",
        "cscwd_router/code/data_prep/synthetic_support.py", "cscwd_router/code/scripts/check_router_dinov2.py",
        "cscwd_router/code/scripts/check_sparse_dinov2.py",
        "dinov2/dinov2/models/vision_transformer.py", "dinov2/dinov2/layers/attention.py", "dinov2/dinov2/layers/block.py",
    ]
    backbone_path = REPO_ROOT / "checkpoints/dinov2_vitb14_pretrain.pth"
    report = {
        "task": "T2", "protocol": "synthetic_support_fit_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv, "config": {name: value for name, value in vars(args).items() if not isinstance(value, Path)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                        "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "precision": "fp32"},
        "synthetic_tests": {"run": synthetic.testsRun, "failures": len(synthetic.failures),
                            "errors": len(synthetic.errors), "skipped": len(synthetic.skipped)},
        "input_checkpoint": {"path": str(backbone_path), "sha256": sha256(backbone_path)},
        "sources": [{"path": str(REPO_ROOT / name), "sha256": sha256(REPO_ROOT / name)} for name in source_names],
        "batch": {"image_shape": list(images.shape), "heads_per_image": [len(boxes) for boxes in batch.bboxes],
                  "target_shape": list(targets.shape), "feature_shape": list(features.shape), "real_data_read": False},
        "fit": {"history": history, "coverage_decreased": coverage_decreased,
                "gradient_l1_at_step2": gradients, "branches_connected": branches_connected,
                "backbone_trainable_parameters": backbone_trainable, "backbone_parameters_with_grad": backbone_gradients},
        "routing_cases": cases,
        "passed": (synthetic.wasSuccessful() and not synthetic.skipped and coverage_decreased and branches_connected
                   and backbone_trainable == 0 and backbone_gradients == 0 and all(case["passed"] for case in cases)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    with args.provenance.open("a") as stream:
        stream.write(f"\n## T2 · {report['created_utc']}\n\n")
        stream.write(f"- code commit: `{report['git_commit']}`\n")
        stream.write(f"- report: `{args.output.resolve()}`; SHA256: `{sha256(args.output)}`; passed: `{report['passed']}`\n")
        stream.write(f"- input: `{backbone_path}`; SHA256: `{report['input_checkpoint']['sha256']}`\n")
        stream.write(f"- synthetic coverage loss: `{history[0]['coverage']}` → `{history[-1]['coverage']}`; steps: `{args.steps}`\n")
    print(f"Report: {args.output}; passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
