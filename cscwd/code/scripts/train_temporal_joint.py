"""R2: smoke-train only the new causal GRU on frozen development features."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cscwd/code"))

from eval_temporal import verify_completed_cache
from prepare_development import artifact, standard_metrics
from stag.cache import ARRAY_FILES, argmax_points, file_sha256, load_cache, write_json
from stag.feature_cache import load_feature_cache
from stag.metrics import TemporalConfig, evaluate_temporal
from stag.state_metrics import evaluate_state_temporal
from stag.temporal_data import iter_track_batches
from stag.temporal_model import CausalJointRefiner, detach_state, joint_loss


def training_inputs(feature_cache, feature_report, r1_report, r1_review):
    r1 = json.loads(Path(r1_report).read_text())
    review = json.loads(Path(r1_review).read_text())
    feature_run = json.loads(Path(feature_report).read_text())
    if (review["status"] != "accepted_for_R2_preparation"
            or review["source_report"]["sha256"] != file_sha256(r1_report)):
        raise ValueError("Training requires the matching accepted R1 review")
    if not feature_run["passed"] or feature_run["mode"] != "cache":
        raise ValueError("Training requires the completed frozen feature cache report")
    feature_cache = Path(feature_cache)
    if file_sha256(feature_cache / "index.json") != feature_run["cache"]["index_sha256"]:
        raise ValueError("Feature index SHA256 mismatch")
    feature_index, features = load_feature_cache(feature_cache)
    for item in feature_index["files"].values():
        print(f"Verifying frozen feature file: {feature_cache / item['path']}", flush=True)
        if file_sha256(feature_cache / item["path"]) != item["sha256"]:
            raise ValueError("Frozen feature file SHA256 mismatch")
    manifest_path = Path(r1["artifacts"]["split_manifest"]["path"])
    manifest_sha = file_sha256(manifest_path)
    if manifest_sha != r1["artifacts"]["split_manifest"]["sha256"]:
        raise ValueError("Development split manifest SHA256 mismatch")
    split = json.loads(manifest_path.read_text())
    cache_dir = r1["cache"]["path"]
    verify_completed_cache(cache_dir, {"passed": r1["passed"], "cache": r1["cache"]})
    prediction_index, arrays = load_cache(cache_dir)
    expected_sources = {"r1_report_sha256": file_sha256(r1_report),
                        "r1_review_sha256": file_sha256(r1_review),
                        "split_manifest_sha256": manifest_sha,
                        "prediction_index_sha256": r1["cache"]["index_sha256"]}
    if (feature_index["sources"] != feature_run["sources"]
            or any(feature_index["sources"][key] != value for key, value in expected_sources.items())):
        raise ValueError("Features and predictions do not share the accepted development sources")
    image_paths = {frame["path"] for track in prediction_index["tracks"] for frame in track["frames"]}
    if set(feature_index["image_paths"]) != image_paths or len(feature_index["image_paths"]) != len(image_paths):
        raise ValueError("Frozen feature image coverage differs from the prediction cache")
    by_id = {track["track_id"]: track for track in prediction_index["tracks"]}
    tracks = {name: [by_id[identity] for identity in part["track_ids"]]
              for name, part in split["partitions"].items()}
    return r1, feature_run, feature_index, features, prediction_index, arrays, tracks, expected_sources


def forward_chunk(model, batch, device, state):
    heatmaps, scores = [], []
    for step in range(batch["valid"].shape[1]):
        heatmap, score, state = model.step(
            batch["features"][:, step].to(device=device, dtype=torch.float32),
            batch["raw_heatmap"][:, step].to(device), batch["raw_inout"][:, step].to(device),
            batch["bboxes"][:, step].to(device), state, batch["reset"][:, step].to(device))
        heatmaps.append(heatmap)
        scores.append(score)
    return torch.stack(heatmaps, dim=1), torch.stack(scores, dim=1), state


def train_epoch(model, optimizer, tracks, arrays, feature_index, features, device,
                clip_frames, batch_size, seed, lambda_state):
    model.train()
    state = None
    totals = {"heatmap_sum": 0.0, "state_sum": 0.0, "valid_frames": 0, "inframe_frames": 0,
              "chunks": 0}
    for batch in iter_track_batches(tracks, arrays, feature_index, features,
                                     clip_frames, batch_size, seed):
        optimizer.zero_grad(set_to_none=True)
        heatmap, score, state = forward_chunk(model, batch, device, state)
        losses = joint_loss(heatmap, score, batch["target_heatmap"].to(device),
                            batch["target_inout"].to(device), batch["valid"].to(device), lambda_state)
        if not torch.isfinite(losses["total"]):
            raise ValueError("Temporal training loss is non-finite")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        state = detach_state(state)
        valid, inside = int(losses["valid_frames"]), int(losses["inframe_frames"])
        totals["heatmap_sum"] += float(losses["heatmap"].detach()) * inside
        totals["state_sum"] += float(losses["state"].detach()) * valid
        totals["valid_frames"] += valid
        totals["inframe_frames"] += inside
        totals["chunks"] += 1
        if totals["chunks"] % 100 == 0:
            print(f"Training chunks={totals['chunks']} frames={totals['valid_frames']}", flush=True)
    map_loss = totals["heatmap_sum"] / max(1, totals["inframe_frames"])
    state_loss = totals["state_sum"] / totals["valid_frames"]
    return {**totals, "heatmap_loss": map_loss, "state_loss": state_loss,
            "total_loss": map_loss + lambda_state * state_loss}


def evaluate_model(model, tracks, arrays, feature_index, features, output_dir,
                   source_frames, device, clip_frames, batch_size, metadata):
    """Carry complete-track state and write compact validation predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    compact_tracks, count = [], 0
    destination = np.full(source_frames, -1, dtype=np.int64)
    for track in tracks:
        stop = count + len(track["frames"])
        compact_tracks.append({**track, "source_global_start": track["start"],
                               "source_global_stop": track["stop"], "start": count, "stop": stop})
        destination[track["start"]:track["stop"]] = np.arange(count, stop)
        count = stop
    shapes = {"heatmap": (count, 64, 64), "inout": (count,), "argmax_xy": (count, 2)}
    predicted = {name: np.lib.format.open_memmap(output_dir / ARRAY_FILES[name], mode="w+",
                    dtype=np.float32, shape=shape) for name, shape in shapes.items()}
    seen = np.zeros(count, dtype=bool)
    state = None
    model.eval()
    with torch.no_grad():
        for batch in iter_track_batches(tracks, arrays, feature_index, features,
                                         clip_frames, batch_size):
            heatmap, score, state = forward_chunk(model, batch, device, state)
            valid = batch["valid"].numpy()
            positions = destination[batch["global_indices"].numpy()[valid]]
            if np.any(positions < 0) or seen[positions].any():
                raise ValueError("Validation frame is outside its manifest or written twice")
            maps = heatmap.cpu().numpy()[valid]
            predicted["heatmap"][positions] = maps
            predicted["inout"][positions] = score.cpu().numpy()[valid]
            predicted["argmax_xy"][positions] = argmax_points(maps)
            seen[positions] = True
    if not seen.all():
        raise ValueError("Validation predictions do not cover every manifest frame")
    for values in predicted.values():
        values.flush()
    files = {name: {"path": ARRAY_FILES[name], "shape": list(shapes[name]), "dtype": "float32",
                    "sha256": file_sha256(output_dir / ARRAY_FILES[name])} for name in predicted}
    index = {"schema_version": 1, "task": "R2_joint_validation", "metadata": metadata,
             "person_frames": count, "track_count": len(compact_tracks), "tracks": compact_tracks,
             "files": files, "passed": True, "checks": {"all_validation_frames_written_once": True}}
    write_json(output_dir / "index.json", index)
    config = TemporalConfig()
    temporal = evaluate_temporal(compact_tracks, predicted["argmax_xy"], predicted["inout"], config)
    state_result = evaluate_state_temporal(compact_tracks, predicted["inout"], config)
    details_path = output_dir / "validation_details.json"
    write_json(details_path, {"joint": {key: temporal[key] for key in ("tracks", "events")},
                               "state": {key: state_result[key] for key in ("tracks", "events")}})
    return {"standard": standard_metrics(compact_tracks, predicted), "temporal": temporal["summary"],
            "state": state_result["summary"], "details": artifact(details_path),
            "cache": {"path": str(output_dir.resolve()), "files": files,
                      "index_sha256": file_sha256(output_dir / "index.json"), "passed": True}}


def run_smoke_training(feature_cache, feature_report, r1_report, r1_review, output_dir,
                       epochs=1, learning_rate=0.0001, lambda_state=0.1, hidden_dim=128,
                       clip_frames=16, batch_size=8, seed=0, device="cuda:0"):
    if epochs < 1:
        raise ValueError("Smoke training requires at least one epoch")
    r1, feature_run, feature_index, features, prediction_index, arrays, tracks, sources = training_inputs(
        feature_cache, feature_report, r1_report, r1_review)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = CausalJointRefiner(feature_index["contract"]["channels"], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {"model": "ordinary_causal_GRU_joint", "feature_channels": model.feature_channels,
              "hidden_dim": hidden_dim, "clip_frames": clip_frames, "batch_size": batch_size,
              "epochs": epochs, "learning_rate": learning_rate, "lambda_state": lambda_state,
              "seed": seed, "optimizer": "AdamW", "weight_decay": 0.01,
              "gradient_clip_norm": 1.0, "device": str(device), "precision": "float32; no_autocast; tf32_disabled",
              "history": "own_predicted_outputs; persistent_slots; detach_only_at_TBPTT_chunks",
              "selection": "none; fixed_smoke_configuration; retain_last_epoch; no_test"}
    write_json(output_dir / "training_config.json", config)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    history = []
    for epoch in range(1, epochs + 1):
        print(f"Smoke epoch {epoch}/{epochs}", flush=True)
        training = train_epoch(model, optimizer, tracks["train"], arrays, feature_index, features,
                               device, clip_frames, batch_size, seed + epoch - 1, lambda_state)
        validation = evaluate_model(model, tracks["validation"], arrays, feature_index, features,
                    output_dir / f"validation_epoch_{epoch:03d}", prediction_index["person_frames"],
                    device, clip_frames, batch_size, {"sources": sources, "config": config, "epoch": epoch})
        checkpoint_path = output_dir / "last.pt"
        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                    "feature_contract": feature_index["contract"], "sources": sources}, checkpoint_path)
        history.append({"epoch": epoch, "training": training, "validation": validation})
        print(f"Epoch {epoch}: train_loss={training['total_loss']:.6f} "
              f"validation={validation['standard']}", flush=True)
    changed = {name: not torch.equal(initial[name], parameter.detach().cpu())
               for name, parameter in model.named_parameters()}
    checks = {"train_frames_match_manifest": all(item["training"]["valid_frames"] ==
                r1["split"]["partitions"]["train"]["counts"]["person_frames"] for item in history),
              "validation_frames_match_manifest": all(item["validation"]["standard"]["person_frames"] ==
                r1["split"]["partitions"]["validation"]["counts"]["person_frames"] for item in history),
              "heatmap_head_updated": any(value for name, value in changed.items() if name.startswith("heatmap_residual")),
              "state_head_updated": any(value for name, value in changed.items() if name.startswith("inout_residual")),
              "only_new_module_parameters": not any(name.startswith("backbone") for name in changed),
              "frozen_features_read_only": not features.flags.writeable,
              "prediction_arrays_read_only": all(not values.flags.writeable for values in arrays.values())}
    if not all(checks.values()):
        raise ValueError(f"R2 smoke training verification failed: {checks}")
    write_json(output_dir / "training_history.json", history)
    return {"task": "R2_joint_smoke", "passed": True, "stage_status": "awaiting_server_result_review",
            "config": config, "sources": sources, "feature_contract": feature_index["contract"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint": artifact(checkpoint_path), "history": history,
            "baseline_validation": r1["baseline"]["validation"], "checks": checks,
            "source_reports": {name: artifact(path) for name, path in (
                ("feature_report", feature_report), ("r1_report", r1_report), ("r1_review", r1_review))},
            "interpretation": "smoke_pipeline_only; not_final_config_selection_or_test_result"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-cache", "feature-report", "r1-report", "r1-review", "output-dir",
                 "report", "summary", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--lambda-state", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_smoke_training(args.feature_cache, args.feature_report, args.r1_report, args.r1_review,
        args.output_dir, args.epochs, args.learning_rate, args.lambda_state, args.hidden_dim,
        args.clip_frames, args.batch_size, args.seed, args.device)
    paths = [Path(__file__), REPO_ROOT / "gazelle/utils.py", *(REPO_ROOT / f"cscwd/code/stag/{name}.py" for name in
             ("temporal_model", "temporal_data", "feature_cache", "metrics", "state_metrics"))]
    report.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
                   "command": shlex.join([sys.executable, *sys.argv]),
                   "code_sha256": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths},
                   "elapsed_seconds": time.perf_counter() - started, "runtime": {
                       "torch": torch.__version__, "numpy": np.__version__,
                       "device_name": torch.cuda.get_device_name(torch.device(args.device))
                           if torch.device(args.device).type == "cuda" else "CPU"}})
    write_json(args.report, report)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    last = report["history"][-1]
    args.summary.write_text("# R2 causal joint-head smoke training\n\n"
        f"- Config: `{json.dumps(report['config'])}`\n- Last epoch: `{last['epoch']}`; no checkpoint selection.\n"
        f"- Validation: `{json.dumps(last['validation']['standard'])}`\n"
        "- Joint/state temporal results and all epoch records are in the JSON report.\n"
        "- This is a development smoke run, not a final method or test result.\n")
    with args.provenance.open("a") as stream:
        stream.write(f"\n## R2 causal joint-head smoke training\n\n- UTC: `{report['created_at_utc']}`; commit: `{report['git_commit']}`.\n"
            f"- Command: `{report['command']}`.\n- Report: `{args.report}`; SHA256: `{file_sha256(args.report)}`.\n"
            f"- Server checkpoint: `{report['checkpoint']['path']}`; SHA256: `{report['checkpoint']['sha256']}`.\n"
            f"- Validation cache: `{last['validation']['cache']['path']}`; index SHA256: `{last['validation']['cache']['index_sha256']}`.\n")
    print(f"R2 joint smoke completed; report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
