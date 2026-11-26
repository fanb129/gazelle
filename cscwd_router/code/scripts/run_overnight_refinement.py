"""Run tonight's AP or unconditional-routing queue and publish small reports."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cscwd_router.code.scripts import run_final_ablation as ablation
from cscwd_router.code.stag.dinov3_run import atomic_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("ap", "unconditional"), required=True)
    for name in ("original-root", "gf-data-path", "vat-data-path", "artifact-root", "report-root", "provenance"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--publish-branch", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--head-lr", type=float, default=1e-2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def replace_arg(command, key, value):
    command[command.index("--" + key) + 1] = str(value)


def commands(args):
    old, weights, reports = map(Path, (args.original_root, args.artifact_root, args.report_root))
    common = ["--provenance", args.provenance, "--workers", str(args.workers), "--seed", str(args.seed),
              "--log-every", "50", "--device", "cuda"] + (["--resume"] if args.resume else [])
    if args.mode == "ap":
        result = []
        for size, batch, source in (("b", 16, "vat_from_gfb_router_k050_joint_seed0"),
                                    ("l", 4, "vat_from_gfl_router_k050_joint_seed0")):
            relative = Path("ap") / f"vit{size}"
            command = [sys.executable, "-u", str(ROOT / "cscwd_router/code/scripts/train_vat_ap.py"),
                "--source-summary", str(old / "cscwd_router/results" / source / "summary.json"),
                "--data-path", args.vat_data_path, "--run-dir", str(weights / relative),
                "--report-dir", str(reports / relative), "--epochs", str(args.epochs),
                "--head-lr", str(args.head_lr), "--batch-size", str(batch), "--eval-batch-size", str(batch),
                "--clip-norm", "1", *common]
            result.append((relative, command))
        return result
    branch_args = ablation.parse_args(["--condition", "unconditional", "--gf-data-path", args.gf_data_path,
        "--vat-data-path", args.vat_data_path, "--initialization-root", str(old),
        "--artifact-root", str(weights), "--report-root", str(reports), "--provenance", args.provenance,
        "--batch-size", "16", "--eval-batch-size", "16", "--grad-accumulation", "2", "--workers", str(args.workers),
        "--support-epochs", "3", "--decoder-epochs", "5", "--vat-epochs", "8", "--seed", str(args.seed),
        *(["--resume"] if args.resume else [])])
    gf, vat = ablation.commands(branch_args)
    # The old scene-only support stage already learned query-independent scores.
    # Head priorities affect hard selection, not that stage's differentiable loss.
    parent_path = old / "cscwd_router/results/final_ablation_seed0/scene_only/gazefollow/vitb/summary.json"
    parent = json.loads(parent_path.read_text())
    support = parent["artifacts"]["support_checkpoint"]
    replace_arg(gf, "init-kind", "support")
    replace_arg(gf, "init-checkpoint", support["path"])
    replace_arg(gf, "init-sha256", support["sha256"])
    atomic_json(reports / "unconditional/inherited_support.json", {
        "source_summary": str(parent_path), "support_checkpoint": support, "inherited_support_epochs": 3,
        "new_decoder_epochs": 5, "new_vat_epochs": 8,
        "removed": ["head_appearance", "absolute_box", "relative_geometry", "head_priority"],
        "retained": ["scene_scoring", "fixed_escape_priority", "decoder_head_box_prompt"],
        "reason": "Reuse scene-only soft-score training; newly adapt decoder after removing head priority."})
    return [(Path("unconditional/gazefollow/vitb"), gf), (Path("unconditional/vat/vitb"), vat)]


def publish(args, relative):
    # Both queues share this lock; only their new result directory is staged.
    lock = Path(args.artifact_root) / "publish.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        reports = Path(args.report_root).resolve()
        git_paths = [str((reports / relative).relative_to(ROOT)),
                     str(Path(args.provenance).resolve().relative_to(ROOT))]
        inherited = reports / "unconditional/inherited_support.json"
        if inherited.exists():
            git_paths.append(str(inherited.relative_to(ROOT)))
        subprocess.run(["git", "add", "--", *git_paths], cwd=ROOT, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
            subprocess.run(["git", "commit", "-m", f"results(cscwd): overnight {relative.as_posix()}"], cwd=ROOT, check=True)
        pushed = subprocess.run(["git", "push", "origin", f"HEAD:{args.publish_branch}"], cwd=ROOT)
        print(f"PUBLISH branch={args.publish_branch} exit_code={pushed.returncode}", flush=True)


def run(args):
    for relative, command in commands(args):
        print("START " + json.dumps(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        publish(args, relative)
    print(f"COMPLETE overnight mode={args.mode}", flush=True)


if __name__ == "__main__":
    run(parse_args())
