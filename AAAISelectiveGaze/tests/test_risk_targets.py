import numpy as np
import pytest

from AAAISelectiveGaze.data.prediction_cache import (
    derive_risk_targets,
    load_prediction_cache,
    make_sample_id,
    save_prediction_cache,
)


def _record(inout=1, probability=0.8):
    final = np.zeros((8, 8), dtype=float)
    final[2, 4] = 1.0
    probe = np.zeros((8, 8), dtype=float)
    probe[3, 4] = 1.0
    return {
        "sample_id": make_sample_id("synthetic", "frame.jpg", 0),
        "dataset": "synthetic",
        "split": "smoke",
        "image_path": "frame.jpg",
        "person_index": 0,
        "final_heatmap": final,
        "probe_heatmaps": {"2": probe, "5": probe},
        "inout_probability": probability,
        "gt_gaze": [[0.5, 0.25]] if inout else [[-1.0, -1.0]],
        "gt_inout": inout,
        "bbox": [0.1, 0.1, 0.2, 0.3],
        "checkpoint_hash": "abc",
    }


def test_risk_targets_are_automatically_derived():
    targets = derive_risk_targets(_record(), failure_l2_threshold=0.15)
    assert targets["localization_l2"] == pytest.approx(0.0)
    assert targets["localization_failure"] == 0
    assert targets["visibility_failure"] == 0
    assert targets["visibility_brier"] == pytest.approx(0.04)


def test_out_of_frame_has_no_localization_target():
    targets = derive_risk_targets(_record(inout=0, probability=0.8))
    assert targets["localization_l2"] is None
    assert targets["localization_failure"] is None
    assert targets["visibility_failure"] == 1


def test_prediction_cache_round_trip(tmp_path):
    path = tmp_path / "predictions.parquet"
    save_prediction_cache([_record()], path)
    loaded = load_prediction_cache(path)
    assert loaded[0]["sample_id"] == _record()["sample_id"]
    assert np.asarray(loaded[0]["final_heatmap"]).shape == (8, 8)


def test_duplicate_sample_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate sample_id"):
        save_prediction_cache([_record(), _record()], tmp_path / "bad.json")
