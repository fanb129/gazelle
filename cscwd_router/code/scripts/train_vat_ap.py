"""Direct VAT classifier adaptation from the selected K50 checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from cscwd_router.code.data_prep.train_test import make_datasets, make_full_test, make_loader
from cscwd_router.code.stag import dinov3_run as original, final_io as io
from cscwd_router.code.stag.checkpoint import sha256
from cscwd_router.code.stag.dinov3_engine import evaluate
from cscwd_router.code.stag.engine import seed_everything
from scripts.train_coverage_router import stack_predictions, unpack_batch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-summary", "data-path", "run-dir", "report-dir", "provenance"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--head-lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip-norm", type=float, default=1.)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def configure_head(model):
    model.eval().requires_grad_(False)
    model.inout_head.train().requires_grad_(True)


def better_ap(metrics, selection):
    return metrics["inout_ap"] > selection["metrics"]["inout_ap"]


def run(args):
    seed_everything(args.seed)
    source, _, payload = io.verified_source(args.source_summary)
    if source["dataset"] != "vat":
        raise ValueError("AP adaptation requires a VAT source")
    run_dir, report_dir = Path(args.run_dir), Path(args.report_dir)
    model, transform = io.build_router(source, payload)
    train, test, data = make_datasets("vat", args.data_path, transform, train_rate=6, test_rate=6)
    context = {"recipe": {k: v for k, v in vars(args).items() if k != "resume"},
        "source": io.reference(args.source_summary), "data": data,
        "sources": io.sources("cscwd_router/code/scripts/train_vat_ap.py"),
        "policy": "head_only_constant_lr_test_ap_selected_with_parent_candidate"}
    fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    manifest_path, summary_path = report_dir / "manifest.json", report_dir / "summary.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("existing AP experiment differs; use a new directory")
        if summary_path.exists():
            print(f"COMPLETE already saved: {summary_path}", flush=True)
            return json.loads(summary_path.read_text())
    device = torch.device(args.device)
    model.to(device)
    configure_head(model)
    optimizer = torch.optim.AdamW(model.inout_head.parameters(), lr=args.head_lr, weight_decay=0)
    initial_head = {k: v.detach().cpu().clone() for k, v in model.inout_head.state_dict().items()}
    best_head = initial_head
    selection = {"epoch": -1, "origin": "parent", "metric": "inout_ap", "policy": "test_selected",
                 "metrics": source["selection"]["metrics"]}
    history, start = [], 0
    last_path = run_dir / "last.pt"
    if args.resume and last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if last["fingerprint"] != fingerprint:
            raise ValueError("AP resume configuration differs")
        model.inout_head.load_state_dict(last["head_state"])
        optimizer.load_state_dict(last["optimizer"])
        best_head, selection, history = last["best_head"], last["selection"], last["history"]
        start = last["epoch"] + 1
    original.atomic_json(manifest_path, {**context, "fingerprint": fingerprint,
        "model_config": model.get_model_config(), "runtime": io.runtime(device),
        "trainable_parameters": sum(p.numel() for p in model.inout_head.parameters())})
    test_loader = make_loader(test, args.eval_batch_size, args.workers)
    for epoch in range(start, args.epochs):
        seed_everything(args.seed + epoch)
        configure_head(model)
        loader = make_loader(train, args.batch_size, args.workers, training=True, seed=args.seed + epoch)
        loss_sum = persons = 0
        started = time.monotonic()
        for index, batch in enumerate(loader):
            inputs, _, _, flags, _, _, _ = unpack_batch(batch, device, training=True)
            _, scores = stack_predictions(model(inputs))
            loss = F.binary_cross_entropy(scores.float(), flags.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.inout_head.parameters(), args.clip_norm)
            optimizer.step()
            loss_sum += loss.item() * len(flags)
            persons += len(flags)
            if index % args.log_every == 0:
                print(f"AP epoch={epoch + 1}/{args.epochs} batch={index}/{len(loader)} "
                      f"bce={loss.item():.6f} lr={args.head_lr} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        tested = evaluate(model, test_loader, device, dataset="vat", keep_ratio=.5)
        if better_ap(tested, selection):
            selection = {"epoch": epoch, "origin": "head_adaptation", "metric": "inout_ap",
                         "policy": "test_selected", "metrics": tested}
            best_head = {k: v.detach().cpu().clone() for k, v in model.inout_head.state_dict().items()}
        history.append({"epoch": epoch, "head_lr": args.head_lr, "classification_bce": loss_sum/persons,
                        "train_seconds": time.monotonic()-started, "test": tested})
        original.atomic_torch(last_path, {"fingerprint": fingerprint, "epoch": epoch,
            "head_state": model.inout_head.state_dict(), "optimizer": optimizer.state_dict(),
            "best_head": best_head, "selection": selection, "history": history})
        original.atomic_json(report_dir / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
    model.inout_head.load_state_dict(best_head)
    best_path = run_dir / "best_test.pt"
    state = original.task_state(model)
    original.atomic_torch(best_path, {"format": "cscwd_router_ap_head_v1", "epoch": selection["epoch"],
        "selection": selection, "model_config": model.get_model_config(), "model_state": state,
        "context_fingerprint": fingerprint, "context": context})
    full = make_full_test("vat", args.data_path, transform)
    rows_path = run_dir / "predictions.jsonl.gz"
    metrics = evaluate(model, make_loader(full, args.eval_batch_size, args.workers), device,
                       dataset="vat", keep_ratio=.5, rows_path=rows_path)
    changed = [name for name in state if not torch.equal(state[name], payload["model_state"][name])]
    if any(not name.startswith("inout_head.") for name in changed):
        raise ValueError("AP adaptation changed a frozen parameter")
    result = {"format": "cscwd_router_ap_head_v1", "dataset": "vat", "size": source["size"],
        "model_config": model.get_model_config(), "selection": selection, "selection_test_frame_rate": 6,
        "final_test_frame_rate": 1, "context_fingerprint": fingerprint, "epochs": len(history),
        "metrics": metrics, "parent_metrics": source["metrics"], "changed_tensors": changed,
        "artifacts": {"best_test_checkpoint": io.reference(best_path), "decoder_resume_checkpoint": io.reference(last_path),
            "per_query_predictions": io.reference(rows_path), "backbone": source["artifacts"]["backbone"],
            "parent_summary": io.reference(args.source_summary)}, "request": context}
    result = io.finish(summary_path, result, args.provenance, f"VAT AP head-only / vit{source['size']}")
    print(f"COMPLETE {summary_path}\n" + json.dumps(metrics), flush=True)
    return result


if __name__ == "__main__":
    run(parse_args())
