"""Review completed GF-to-VAT artifacts on CPU, without a model forward."""

import argparse
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import average_precision_score
import torch

from cscwd_router.code.data_prep.train_test import make_datasets, make_full_test, query_identifier
from cscwd_router.code.scripts import run_router_efficiency as queue
from cscwd_router.code.stag.checkpoint import sha256, state_sha256
from cscwd_router.code.stag.dinov3_sources import load_source
from cscwd_router.code.stag.dinov3_transfer import is_inout, parameter_group, source_hashes


def review(args):
    torch.set_num_threads(2)
    summary, manifest, best = load_source(args.summary)
    assert summary["completed"] and summary["dataset"] == "vat"
    context = {k: v for k, v in manifest.items() if k not in ("fingerprint", "runtime")}
    assert hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest() == manifest["fingerprint"]
    sources = source_hashes()
    assert manifest["sources"] == sources
    artifacts = summary["artifacts"]
    for item in artifacts.values():
        assert sha256(item["path"]) == item["sha256"], item["path"]
    for split in ("train", "test"):
        item = manifest["data"][split + "_annotation"]
        assert sha256(item["path"]) == item["sha256"]
    gf, _, gf_best = load_source(artifacts["gf_summary"]["path"], allow_closed_gf=True)
    initial = torch.load(artifacts["initial_checkpoint"]["path"], map_location="cpu", weights_only=False)
    last = torch.load(artifacts["decoder_resume_checkpoint"]["path"], map_location="cpu", weights_only=False)
    init_state = initial["model_state"]
    shared = gf_best["model_state"]
    assert set(shared) == {k for k in init_state if not is_inout(k)}
    assert all(torch.equal(v, init_state[k]) for k, v in shared.items())
    initialization = summary["initialization"]
    digest_groups = {
        "gf_router_sha256": {k: v for k, v in shared.items() if k.startswith("router.")},
        "gf_decoder_sha256": {k: v for k, v in shared.items() if not k.startswith("router.")},
        "new_inout_sha256": {k: v for k, v in init_state.items() if is_inout(k)},
        "initial_task_sha256": init_state,
    }
    for key, state in digest_groups.items():
        assert state_sha256(state) == initialization[key]
    history = json.loads(Path(artifacts["history"]["path"]).read_text())
    assert history == last["history"]
    assert [row["epoch"] for row in history] == list(range(summary["epochs"]))
    assert last["epoch"] == len(history) - 1
    selected = min(history, key=lambda row: (row["test"]["l2"], -row["test"]["auc"], row["epoch"]))
    assert selected["epoch"] == summary["selection"]["epoch"] == best["epoch"]
    assert selected["test"] == summary["selection"]["metrics"]
    assert last["selection"] == summary["selection"]
    assert state_sha256(last["best_model_state"]) == state_sha256(best["model_state"])
    epoch_checks = []
    for row in history:
        epoch = row["epoch"]
        payload = torch.load(artifacts[f"epoch_{epoch}"]["path"], map_location="cpu", weights_only=False)
        assert payload["epoch"] == epoch and payload["test_metrics"] == row["test"]
        assert payload["context_fingerprint"] == summary["context_fingerprint"]
        assert payload["model_config"] == summary["model_config"]
        state = payload["model_state"]
        assert set(state) == set(init_state) and not any(k.startswith("backbone.") for k in state)
        changed = {group: sum(not torch.equal(v, init_state[k]) for k, v in state.items()
                              if parameter_group(k) == group) for group in ("router", "decoder", "inout")}
        active = epoch >= summary["router_warmup_epochs"]
        assert row["train"]["router_active"] == active
        assert row["train"]["trainable_parameters"]["backbone"] == 0
        assert bool(changed["router"]) == active and changed["decoder"] and changed["inout"]
        assert row["train"]["train_keep_ratio"] == 0.5
        if epoch == selected["epoch"]:
            assert state_sha256(state) == state_sha256(best["model_state"])
        epoch_checks.append({"epoch_one_based": epoch + 1, "changed_tensors_vs_initial": changed})
    steps = {group["name"]: sorted({int(last["optimizer"]["state"][p]["step"]) for p in group["params"]})
             for group in last["optimizer"]["param_groups"]}
    for name, observed in steps.items():
        expected = sum(row["train"]["optimizer_steps"] for row in history
                       if name != "router" or row["train"]["router_active"])
        assert observed == [expected]
    recipe = manifest["recipe"]
    assert recipe["seed"] == 0 and summary["selection_policy"] == "test_selected"
    train, test, data = make_datasets("vat", recipe["data_path"], None,
                                    train_rate=recipe["train_frame_rate"], test_rate=recipe["test_frame_rate"])
    assert data == manifest["data"]
    assert all(row["train"]["persons"] == len(train) and row["test"]["persons"] == len(test) for row in history)
    full = make_full_test("vat", recipe["data_path"], None)
    with gzip.open(artifacts["per_query_predictions"]["path"], "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == len(full)
    for index, row in enumerate(rows):
        image_index, head_index = full.data_idxs[index]
        head = full.data[image_index]["heads"][head_index]
        assert row["sample_index"] == index
        assert {k: row[k] for k in ("path", "head_index")} == query_identifier(full, index)
        assert (row["gazex"], row["gazey"], row["in_frame"]) == (head["gazex_norm"], head["gazey_norm"], head["inout"])
    means = np.asarray([[r["auc"], r["l2"]] for r in rows if r["in_frame"]]).mean(0).tolist()
    aggregated = dict(zip(("auc", "l2"), means))
    aggregated.update(inout_ap=float(average_precision_score([r["in_frame"] for r in rows], [r["inout_score"] for r in rows])),
                      persons=len(rows), inframe_persons=sum(r["in_frame"] for r in rows))
    differences = {key: value - summary["metrics"][key] for key, value in aggregated.items()}
    assert all(value == 0 for value in differences.values())
    preflight = queue.parse_args(["--summaries", str(args.summary), "--data-path", recipe["data_path"],
        "--official-checkpoint-dir", str(args.official_checkpoint_dir), "--report-root", str(args.efficiency_report_root),
        "--provenance", str(args.provenance), "--batches", "1", "8", "--precision", "fp16",
        "--warmup", "50", "--latency-iters", "200", "--throughput-iters", "500", "--repeats", "5",
        "--seed", "0", "--check-only"])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        queue.run(preflight)
    cells = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(cells) == 4
    proof = {"task": "E2-TL_server_review_and_E3L_CPU_preflight", "passed": True,
        "scope": "CPU hashes, tensors, selection, exported query aggregation and benchmark input preflight; no model forward or GPU timing.",
        "review_script_sha256": sha256(__file__), "context_fingerprint": manifest["fingerprint"],
        "report_sha256": {p.name: sha256(p) for p in sorted(args.summary.parent.glob("*.json"))},
        "source_sha256": sources, "artifacts": artifacts,
        "artifact_paths_sha256_verified": len(artifacts), "shared_GF_tensors_identical": len(shared),
        "new_inout_tensors": len(init_state) - len(shared), "gf_source_status": summary["gf_source_status"],
        "epoch_tensor_checks": epoch_checks, "optimizer_internal_steps": steps,
        "selected_epoch_one_based": selected["epoch"] + 1,
        "selected_router_unchanged_from_GF": all(torch.equal(v, shared[k]) for k, v in best["model_state"].items() if k.startswith("router.")),
        "metrics": summary["metrics"], "query_identity_and_annotation_matches": len(rows),
        "aggregate_differences": differences, "efficiency_preflight": cells,
        "efficiency_report_root": str(args.efficiency_report_root)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "selected_epoch_one_based": selected["epoch"] + 1,
                      "metrics": aggregated, "preflight_cells": len(cells), "output": str(args.output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "official-checkpoint-dir", "efficiency-report-root", "provenance", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    review(parser.parse_args())
