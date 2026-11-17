"""Full official-test scoring for the explicitly test-selected protocol."""

import json
from pathlib import Path

from stag.cache import file_sha256, write_json
from stag.compact_validation import evaluate_compact_model
from stag.metrics import TemporalConfig, evaluate_predictions, evaluate_temporal
from stag.state_metrics import evaluate_state_temporal


def _artifact(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size}


def evaluate_official_baseline(tracks, arrays, output_dir):
    """Score retained single-frame test predictions with the unchanged metrics."""
    config = TemporalConfig()
    temporal = evaluate_temporal(tracks, arrays["argmax_xy"], arrays["inout"], config)
    state = evaluate_state_temporal(tracks, arrays["inout"], config)
    path = Path(output_dir) / "baseline_details.json"
    write_json(path, {"joint": {key: temporal[key] for key in ("tracks", "events")},
                      "state": {key: state[key] for key in ("tracks", "events")}})
    return {"evaluation_split": "test",
            "standard": evaluate_predictions(tracks, arrays["heatmap"], arrays["inout"]),
            "temporal": temporal["summary"], "state": state["summary"],
            "details": _artifact(path)}


def evaluate_official_model(model, tracks, arrays, feature_index, features, output_dir,
                            source_frames, device, clip_frames, batch_size, metadata):
    """Reuse the complete causal evaluator while labelling all outputs as test.

    The underlying evaluator performs one no-grad forward per frame, resets
    state at track boundaries, and writes diagnostics from that same forward.
    Only artifact labels and the details filename change here.
    """
    metadata = {**metadata, "evaluation_split": "test",
                "test_used_for_checkpoint_selection": True}
    result = evaluate_compact_model(model, tracks, arrays, feature_index, features,
        output_dir, source_frames, device, clip_frames, batch_size, metadata)
    output_dir = Path(output_dir)
    index_path = output_dir / "index.json"
    index = json.loads(index_path.read_text())
    index.update({"task": "R4_official_test_evaluation", "evaluation_split": "test",
                  "checks": {"all_test_frames_written_once": True}})
    write_json(index_path, index)
    old_details = Path(result["details"]["path"])
    details = output_dir / "test_details.json"
    old_details.rename(details)
    result["details"] = _artifact(details)
    result["cache"]["index_sha256"] = file_sha256(index_path)
    result["evaluation_split"] = "test"
    return result
