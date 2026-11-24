"""Immutable source references, resumable result cells and shared provenance."""

import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

from cscwd_router.code.stag import dinov3_run as original
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag.conditional import get_conditional_model
from cscwd_router.code.stag.dinov3_sources import load_source

ROOT = original.ROOT


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def sources(*names):
    base = ("cscwd_router/code/stag/final_io.py", "cscwd_router/code/stag/conditional.py")
    return {**original.source_hashes(), **{name: sha256(ROOT / name) for name in (*base, *names)}}


def verified_source(path, allow_closed_gf=False):
    summary, manifest, payload = load_source(path, allow_closed_gf=allow_closed_gf)
    factory_weight = ROOT / "checkpoints" / f"dinov3_vit{summary['size']}16_pretrain.pth"
    if sha256(factory_weight) != summary["artifacts"]["backbone"]["sha256"]:
        raise ValueError("factory backbone differs from the selected source")
    return summary, manifest, payload


def build_router(summary, payload, keep_ratio=.5):
    config = dict(summary["model_config"])
    name = config.pop("model")
    condition = config.pop("routing_condition", "full")
    model, transform = get_conditional_model(name, condition=condition, **config)
    original.restore_task(model, payload["model_state"])
    if model.get_model_config() != summary["model_config"]:
        raise ValueError("reconstructed model differs from its selected source")
    model.router.keep_ratio = keep_ratio
    return model.eval().requires_grad_(False), transform


def existing(path, request, resume):
    path = Path(path)
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if not resume or not saved.get("completed") or saved["request"] != request:
        raise ValueError(f"existing result differs or needs --resume: {path}")
    for item in saved.get("artifacts", {}).values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"saved artifact SHA256 mismatch: {item['path']}")
    return saved


def runtime(device):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}


def finish(path, result, provenance, label):
    result = {**result, "completed": True}
    original.atomic_json(Path(path), result)
    Path(provenance).parent.mkdir(parents=True, exist_ok=True)
    with Path(provenance).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f"\n## {label}\n\n- report: `{Path(path).resolve()}`; SHA256: `{sha256(path)}`\n")
        for name, item in result.get("artifacts", {}).items():
            handle.write(f"- {name}: `{item['path']}`; SHA256: `{item['sha256']}`\n")
    return result
