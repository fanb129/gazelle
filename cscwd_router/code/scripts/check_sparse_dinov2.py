"""T1 checkpoint checks on synthetic inputs; this is not a gaze evaluation.

Run from the repository root, where the official GazeLLE loader expects the
local ``dinov2/`` checkout and ``checkpoints/`` directory. No dataset is read.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import timm
import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dinov2"))

from gazelle.backbone_v0_dinov2 import DinoV2Backbone
from gazelle.model_v0_dinov2 import GazeLLE
from cscwd_router.code.stag.backbone_sparse_dinov2 import RoutedDinoV2Backbone


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    return {
        "equal": torch.equal(actual, expected),
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_abs_diff": (actual.float() - expected.float()).abs().max().item(),
        "mismatched_values": (actual != expected).sum().item(),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
    }


def check_case(official, routed, decoder, *, batch_size, precision, device, image_size, route_after_block):
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    # Synthetic boxes exercise both one and multiple decoder queries per image.
    boxes = [[(0.10, 0.10, 0.25, 0.30)] + ([(0.65, 0.10, 0.85, 0.30)] if b % 2 else [])
             for b in range(batch_size)]
    inputs = {"images": images, "bboxes": boxes}
    enabled = precision == "fp16"
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled, cache_enabled=False):
        expected_patches = official.model.forward_features(images)["x_norm_patchtokens"]
        state = routed.forward_prefix(images, route_after_block)
        indices = torch.arange(state.height * state.width, device=device).expand(batch_size, -1)
        actual_map = routed.forward_suffix(state, indices)
        backbone_check = compare(actual_map.flatten(2).transpose(1, 2), expected_patches)
        decoder.backbone = official
        expected_predictions = decoder(inputs)["heatmap"]
        decoder.backbone = routed
        actual_predictions = decoder(inputs)["heatmap"]
        decoder.backbone = official
        decoder_check = compare(torch.cat(actual_predictions), torch.cat(expected_predictions))

        # Check actual half-budget execution and the unchanged early-exit fill.
        sparse_lengths = []
        hook = routed.model.blocks[route_after_block + 1].register_forward_pre_hook(
            lambda _module, args: sparse_lengths.append(args[0].shape[1])
        )
        sparse_indices = torch.arange(0, state.height * state.width, 2, device=device).expand(batch_size, -1)
        sparse_map = routed.forward_suffix(state, sparse_indices)
        hook.remove()
        early = routed.tokens_to_map(state.tokens, state.height, state.width).flatten(2)
        fill_check = compare(sparse_map.flatten(2)[:, :, 1::2], early[:, :, 1::2])

    passed = all(check["equal"] and check["finite"] for check in (backbone_check, decoder_check, fill_check))
    expected_suffix_length = sparse_indices.shape[1] + routed.prefix_token_count
    return {
        "batch_size": batch_size,
        "precision": precision,
        "image_size": [image_size, image_size],
        "route_after_block": route_after_block,
        "k100_backbone": backbone_check,
        "k100_official_decoder": decoder_check,
        "k50_early_exit_fill": fill_check,
        "k50_suffix_lengths": sparse_lengths,
        "expected_k50_suffix_length": expected_suffix_length,
        "passed": passed and sparse_lengths == [expected_suffix_length],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--precisions", choices=("fp32", "fp16"), nargs="+", default=["fp32", "fp16"])
    parser.add_argument("--image-size", type=int, choices=(448,), default=448)
    parser.add_argument("--route-after-block", type=int, choices=(5,), default=5)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()

    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"run from the repository root: {REPO_ROOT}")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    suite = unittest.defaultTestLoader.loadTestsFromName("cscwd_router.code.stag.test_backbone_sparse_dinov2")
    synthetic = unittest.TextTestRunner(verbosity=2).run(suite)
    # Test fixtures seed their own small models; restore the experiment seed.
    torch.manual_seed(args.seed)

    official = DinoV2Backbone("dinov2_vitb14").eval().requires_grad_(False)
    routed = RoutedDinoV2Backbone(model=official.model).eval().requires_grad_(False)
    decoder = GazeLLE(official).eval().requires_grad_(False)
    decoder_state = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=True)
    incompatible = decoder.load_state_dict(decoder_state, strict=False)
    missing_decoder = [key for key in incompatible.missing_keys if not key.startswith("backbone.")]
    if missing_decoder or incompatible.unexpected_keys:
        raise ValueError(f"official decoder checkpoint mismatch: missing={missing_decoder}, unexpected={incompatible.unexpected_keys}")
    decoder.to(device)
    routed.to(device)

    cases = []
    for precision in args.precisions:
        for batch_size in args.batch_sizes:
            case = check_case(official, routed, decoder, batch_size=batch_size, precision=precision,
                              device=device, image_size=args.image_size, route_after_block=args.route_after_block)
            cases.append(case)
            print(json.dumps(case), flush=True)

    input_paths = [REPO_ROOT / "checkpoints/dinov2_vitb14_pretrain.pth", args.decoder_checkpoint.resolve()]
    source_paths = [
        REPO_ROOT / "gazelle/backbone_v0_dinov2.py",
        REPO_ROOT / "gazelle/model_v0_dinov2.py",
        REPO_ROOT / "dinov2/dinov2/models/vision_transformer.py",
        REPO_ROOT / "dinov2/dinov2/layers/block.py",
        REPO_ROOT / "dinov2/dinov2/layers/attention.py",
        REPO_ROOT / "cscwd_router/code/stag/backbone_sparse_dinov2.py",
        REPO_ROOT / "cscwd_router/code/stag/test_backbone_sparse_dinov2.py",
        Path(__file__).resolve(),
    ]
    report = {
        "task": "T1", "protocol": "synthetic_checkpoint_equivalence_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "timm": timm.__version__,
                        "device": str(device), "cuda": torch.version.cuda,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "attention_class": type(official.model.blocks[0].attn).__name__,
                        "tf32": False, "autocast_cache_enabled": False},
        "seed": args.seed,
        "synthetic_tests": {"run": synthetic.testsRun, "failures": len(synthetic.failures),
                            "errors": len(synthetic.errors), "skipped": len(synthetic.skipped)},
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in input_paths],
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "cases": cases,
        "passed": synthetic.wasSuccessful() and not synthetic.skipped and all(case["passed"] for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    with args.provenance.open("a") as stream:
        stream.write(f"\n## T1 · {report['created_utc']}\n\n")
        stream.write(f"- code commit: `{report['git_commit']}`; dirty: `{report['git_dirty']}`\n")
        stream.write(f"- report: `{args.output.resolve()}`; SHA256: `{sha256(args.output)}`; passed: `{report['passed']}`\n")
        for artifact in report["inputs"]:
            stream.write(f"- input: `{artifact['path']}`; SHA256: `{artifact['sha256']}`\n")
    print(f"Report: {args.output}; passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
