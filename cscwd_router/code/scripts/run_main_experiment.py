"""T4: fixed full-train -> one test per checkpoint -> six efficiency cells.

Launch manually on the server with one CUDA_VISIBLE_DEVICES entry. Completed
official evaluations are verified and reused on --resume, never overwritten.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dinov2"))

from cscwd_router.code.scripts.train_router_dinov2 import SOURCE_NAMES
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag.main_results import AUC_DROP_LIMIT, B8_SPEEDUP_MIN, BATCHES, ORDERS, render_comparison, summarize_results
from cscwd_router.code.stag.pipeline import checkpoint_ref, write_json


STAGES = ("dense_official", "sparse_decoder", "matched_dense")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t3-summary", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--router-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--budget-weight", type=float, default=0.05)
    parser.add_argument("--router-hidden-dim", type=int, default=256)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--escape-tokens", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3106)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--latency-iters", type=int, default=200)
    parser.add_argument("--throughput-iters", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def training_command(args, *, resume=False):
    command = [sys.executable, "-u", str(ROOT / "cscwd_router/code/scripts/train_router_dinov2.py"),
               "--protocol", "full_train", "--data-path", str(args.data_path),
               "--decoder-checkpoint", str(args.decoder_checkpoint), "--run-dir", str(args.run_dir),
               "--report-dir", str(args.report_dir / "training"), "--provenance", str(args.provenance),
               "--device", "cuda", "--batch-size", str(args.train_batch_size), "--eval-batch-size", str(args.eval_batch_size),
               "--workers", str(args.workers), "--router-lr", str(args.router_lr), "--decoder-lr", str(args.decoder_lr),
               "--weight-decay", str(args.weight_decay), "--budget-weight", str(args.budget_weight),
               "--seed", str(args.seed), "--split-seed", str(args.seed), "--val-fraction", "0",
               "--router-hidden-dim", str(args.router_hidden_dim), "--router-temperature", str(args.router_temperature),
               "--escape-tokens", str(args.escape_tokens), "--log-every", str(args.log_every)]
    if resume:
        command.append("--resume")
    return command


def evaluation_command(args, stage, output):
    return [sys.executable, "-u", str(ROOT / "cscwd_router/code/scripts/eval_router_dinov2.py"),
            "--checkpoint", str(args.run_dir / stage / "final.pt"), "--split", "official_test",
            "--data-path", str(args.data_path), "--device", "cuda", "--batch-size", str(args.eval_batch_size),
            "--workers", str(args.workers), "--output", str(output), "--provenance", str(args.provenance)]


def benchmark_command(args, batch, order, output):
    return [sys.executable, "-u", str(ROOT / "cscwd_router/code/scripts/benchmark_router_dinov2.py"),
            "--dense-checkpoint", str(args.run_dir / "dense_official/final.pt"),
            "--sparse-checkpoint", str(args.run_dir / "sparse_decoder/final.pt"), "--data-path", str(args.data_path),
            "--batch-size", str(batch), "--order", order, "--precision", "fp16", "--warmup", str(args.warmup),
            "--latency-iters", str(args.latency_iters), "--throughput-iters", str(args.throughput_iters),
            "--repeats", str(args.repeats), "--seed", str(args.seed), "--output", str(output)]


def cached_evaluation_matches(result, *, stage, checkpoint_sha, training_signature, annotation_sha, batch_size):
    return (result["stage"] == stage and result["split"] == "official_test"
            and result["checkpoint"]["sha256"] == checkpoint_sha and result["training_signature"] == training_signature
            and result["data"]["annotation_sha256"] == annotation_sha and result["batch_size"] == batch_size)


def training_is_finalized(result):
    # The trainer writes summary once before adding the actual checkpoint hashes.
    return result.get("completed") and result.get("passed") and bool(result.get("artifacts"))


def run_command(command, journal):
    print("RUN " + shlex.join(command), flush=True)
    journal.append(command)
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv=None):
    args = parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from the repository root: {ROOT}")
    t3 = json.loads(args.t3_summary.read_text())
    if not t3["passed"] or not all(t3["engineering_checks"].values()) or t3["protocol"] != "synthetic":
        raise ValueError("T4 requires the passed T3 synthetic training/checkpoint smoke")
    for name, digest in t3["context"]["sources"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"training source changed since T3: {name}")
    weights = {"backbone": checkpoint_ref(ROOT / "checkpoints/dinov2_vitb14_pretrain.pth"),
               "official_decoder": checkpoint_ref(args.decoder_checkpoint)}
    if any(weights[name]["sha256"] != t3["context"]["external_weights"][name]["sha256"] for name in weights):
        raise ValueError("T4 inputs must match the official weights verified in T3")
    names = (*SOURCE_NAMES, "cscwd_router/code/scripts/run_main_experiment.py",
             "cscwd_router/code/scripts/benchmark_router_dinov2.py", "cscwd_router/code/stag/main_results.py",
             "cscwd_router/code/stag/test_main_experiment.py")
    sources = {name: sha256(ROOT / name) for name in names}
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items() if name != "resume"}
    identity = {"config": config, "sources": sources, "weights": weights, "t3_sha256": sha256(args.t3_summary),
                "train_annotation_sha256": sha256(args.data_path / "train_preprocessed.json"),
                "decision_rule": {"auc_drop_limit": AUC_DROP_LIMIT, "b8_latency_speedup_min": B8_SPEEDUP_MIN,
                                  "accuracy_reference": "official dense", "order_aggregation": "geometric mean of ratios"},
                "efficiency_protocol": {"batches": list(BATCHES), "orders": list(ORDERS), "precision": "fp16",
                                        "autocast_cache_enabled": False, "inference_mode": True}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.report_dir / "manifest.json"
    if args.resume:
        if json.loads(manifest.read_text())["signature"] != signature:
            raise ValueError("T4 resume requires unchanged source, data, weights and settings")
    else:
        with manifest.open("x") as stream:
            stream.write(json.dumps({**identity, "signature": signature, "created_utc": datetime.now(timezone.utc).isoformat(),
                                     "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}, indent=2) + "\n")
    final_path = args.report_dir / "comparison.json"
    if args.resume and final_path.exists():
        result = json.loads(final_path.read_text())
        if result["signature"] != signature or any(sha256(item["path"]) != item["sha256"] for item in result["evidence"]):
            raise ValueError("completed T4 evidence no longer matches its recorded hashes")
        print(json.dumps(result["gate"]), flush=True)
        return 0 if result["gate"]["passed"] else 2

    commands = []
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "dinov2"), str(ROOT), env.get("PYTHONPATH", "")))
    subprocess.run([sys.executable, "-m", "unittest", "-v", "cscwd_router.code.stag.test_main_experiment"], cwd=ROOT, env=env, check=True)
    training_path = args.report_dir / "training/summary.json"
    ready = args.resume and training_path.exists() and training_is_finalized(json.loads(training_path.read_text()))
    if not ready:
        run_command(training_command(args, resume=args.resume and args.run_dir.exists()), commands)
    training = json.loads(training_path.read_text())
    if (not training["passed"] or training["protocol"] != "full_train" or training["context"]["data"]["official_test_accessed"]
            or training["context"]["data"]["validation_persons"] != 0
            or training["context"]["data"]["annotation_sha256"] != identity["train_annotation_sha256"]):
        raise ValueError("T4 training must complete the fixed full-train/no-eval recipe")
    for item in training["artifacts"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"training checkpoint changed: {item['path']}")

    # Official annotation is first opened only after all training has finished.
    test_sha = sha256(args.data_path / "test_preprocessed.json")
    accuracy, evidence = {}, [checkpoint_ref(training_path)]
    for stage in STAGES:
        output = args.report_dir / "accuracy" / f"{stage}.json"
        checkpoint_sha = sha256(args.run_dir / stage / "final.pt")
        if not (args.resume and output.exists()):
            run_command(evaluation_command(args, stage, output), commands)
        result = json.loads(output.read_text())
        if not cached_evaluation_matches(result, stage=stage, checkpoint_sha=checkpoint_sha,
                                         training_signature=training["context"]["signature"], annotation_sha=test_sha,
                                         batch_size=args.eval_batch_size):
            raise ValueError(f"official evaluation does not match fixed inputs: {stage}")
        accuracy[stage] = result
        evidence.append(checkpoint_ref(output))
    benchmarks = []
    expected_settings = {"warmup": args.warmup, "latency_iters": args.latency_iters,
                         "throughput_iters": args.throughput_iters, "repeats": args.repeats}
    for batch in BATCHES:
        for order in ORDERS:
            output = args.report_dir / "efficiency" / f"b{batch}_{order}.json"
            if not (args.resume and output.exists()):
                run_command(benchmark_command(args, batch, order, output), commands)
            result = json.loads(output.read_text())
            if result["batch_size"] != batch or result["order"] != order or any(result["settings"][key] != value for key, value in expected_settings.items()):
                raise ValueError("cached efficiency cell uses different settings")
            benchmarks.append(result)
            evidence.append(checkpoint_ref(output))
    result = summarize_results(accuracy, benchmarks)
    result.update({"signature": signature, "completed": True, "evidence": evidence,
                   "commands_executed_this_invocation": commands, "training_signature": training["context"]["signature"]})
    write_json(final_path, result)
    markdown_path = args.report_dir / "comparison.md"
    markdown_path.write_text(render_comparison(result))
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    with args.provenance.open("a") as stream:
        stream.write(f"\n## T4 · {datetime.now(timezone.utc).isoformat()}\n\n")
        stream.write(f"- comparison: `{final_path.resolve()}`; SHA256: `{sha256(final_path)}`; gate passed: `{result['gate']['passed']}`\n")
        stream.write(f"- table: `{markdown_path.resolve()}`; SHA256: `{sha256(markdown_path)}`\n")
        for item in evidence:
            stream.write(f"- evidence: `{item['path']}`; SHA256: `{item['sha256']}`\n")
    print(json.dumps(result["gate"]), flush=True)
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
