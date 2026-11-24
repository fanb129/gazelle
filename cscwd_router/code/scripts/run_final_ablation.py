"""One GPU: train a conditional GF-B branch, then transfer that branch to VAT-B."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cscwd_router.code.stag.conditional import CONDITIONS
from cscwd_router.code.stag.dinov3_initializers import BACKBONE_SHA, INITIALIZERS
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.data_prep.train_test import make_datasets


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=tuple(k for k in CONDITIONS if k != "full"), required=True)
    for name in ("gf-data-path", "vat-data-path", "artifact-root", "report-root", "provenance", "initialization-root"):
        p.add_argument("--" + name, type=Path, required=True)
    for name, default in (("workers", 8), ("batch-size", 16), ("eval-batch-size", 16), ("grad-accumulation", 2),
                          ("support-epochs", 3), ("decoder-epochs", 5), ("vat-epochs", 8), ("seed", 0)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-only", action="store_true")
    return p.parse_args(argv)


def commands(args, python=sys.executable):
    init = INITIALIZERS["gazefollow", "b"]
    script = str(ROOT / "cscwd_router/code/scripts/train_conditioned_router.py")
    reports, weights = args.report_root / args.condition, args.artifact_root / args.condition
    common = ["--provenance", str(args.provenance), "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size), "--grad-accumulation", str(args.grad_accumulation),
        "--workers", str(args.workers), "--seed", str(args.seed), "--log-every", "50", "--device", "cuda"]
    if args.resume:
        common.append("--resume")
    gf = [python, "-u", script, "--phase", "gf", "--condition", args.condition,
        "--dataset", "gazefollow", "--size", "b", "--data-path", str(args.gf_data_path),
        "--init-checkpoint", str(args.initialization_root / init["path"]), "--init-sha256", init["sha256"],
        "--init-kind", "decoder", "--backbone-sha256", BACKBONE_SHA["b"],
        "--run-dir", str(weights / "gazefollow/vitb"), "--report-dir", str(reports / "gazefollow/vitb"),
        "--support-epochs", str(args.support_epochs), "--decoder-epochs", str(args.decoder_epochs),
        "--keep-ratio", "0.5", "--router-lr", "0.001", "--decoder-lr", "0.0001", "--inout-lr", "0.001",
        "--weight-decay", "0", "--clip-norm", "1", "--budget-weight", "0.05", "--router-hidden-dim", "256",
        "--router-temperature", "1", "--escape-tokens", "8", "--train-frame-rate", "6", "--test-frame-rate", "6", *common]
    vat = [python, "-u", script, "--phase", "vat", "--condition", args.condition,
        "--gf-summary", str(reports / "gazefollow/vitb/summary.json"), "--data-path", str(args.vat_data_path),
        "--run-dir", str(weights / "vat/vitb"), "--report-dir", str(reports / "vat/vitb"),
        "--epochs", str(args.vat_epochs), "--router-warmup-epochs", "1", "--router-lr", "0.0001",
        "--decoder-lr", "0.00001", "--inout-lr", "0.001", "--router-loss-weight", "1", "--budget-weight", "0.05",
        "--clip-norm", "1", "--train-frame-rate", "6", "--test-frame-rate", "6", *common]
    return gf, vat


def preflight(args):
    if args.seed != 0:
        raise ValueError("the final experiment uses seed 0")
    init = INITIALIZERS["gazefollow", "b"]
    if sha256(args.initialization_root / init["path"]) != init["sha256"]:
        raise ValueError("GF decoder initialization SHA256 mismatch")
    if sha256(ROOT / "checkpoints/dinov3_vitb16_pretrain.pth") != BACKBONE_SHA["b"]:
        raise ValueError("DINOv3-B backbone SHA256 mismatch")
    data = {}
    for name, path in (("gazefollow", args.gf_data_path), ("vat", args.vat_data_path)):
        _, _, data[name] = make_datasets(name, path, None)
    return {"condition": args.condition, "seed": args.seed, "data": data,
            "commands": commands(args), "gpu_forward_executed": False}


def run(args):
    checked = preflight(args)
    if args.check_only:
        print(json.dumps(checked, indent=2), flush=True)
        return checked
    for command in commands(args):
        print("START " + json.dumps(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"COMPLETE conditional branch {args.condition}", flush=True)


if __name__ == "__main__":
    run(parse_args())
