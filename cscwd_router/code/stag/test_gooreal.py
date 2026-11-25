"""Check coordinate normalization, AUC ties and raw GT-box auditing."""

import json
import pickle

import numpy as np
import pytest

from cscwd_router.code.data_prep.gooreal import audited_records
from cscwd_router.code.stag.gooreal import query_metrics, visualization_indices
from data_prep.preprocess_gooreal import convert_record


def test_benchmark_coordinates_and_auc():
    heat = np.zeros((5, 5), dtype=np.float32)
    heat[3, 2] = 1
    result = query_metrics(heat, .4, .6)
    assert result == {"auc": 1., "l2": 0., "prediction": [.4, .6]}
    assert query_metrics(np.ones((64, 64)), .4, .6)["auc"] == .5
    # Normalize by the heatmap width, rather than width-1 or a half-pixel shift.
    heat = np.zeros((64, 64), dtype=np.float32)
    heat[48, 32] = 1
    assert query_metrics(heat, .5, .75)["l2"] == 0


def test_audit_uses_annotation_canvas_and_real_human_box(tmp_path):
    raw = {"filename": "73\\cam0\\a.jpg", "width": 1920, "height": 1080,
           "gaze_cx": 359, "gaze_cy": 360, "hx": 300, "hy": 200,
           "ann": {"bboxes": [[266, 160, 335, 261]], "labels": [25]}}
    converted = convert_record(raw, 0, "finalrealdatasetImgsV3")
    assert not converted.get("gooreal_warnings")
    (tmp_path / "testrealhumansNew.pickle").write_bytes(pickle.dumps([raw]))
    path = tmp_path / "gooreal_test_preprocessed.json"
    path.write_text(json.dumps([converted]))
    records, audit = audited_records(tmp_path)
    assert audit["annotated_human_boxes"] == 1
    assert records[0]["heads"][0]["gazey_norm"] == [.75]
    converted["heads"][0]["gazey_norm"] = [360/1080]
    path.write_text(json.dumps([converted]))
    with pytest.raises(AssertionError):
        audited_records(tmp_path)
    converted["heads"][0]["gazey_norm"] = [.75]
    converted["heads"][0]["gooreal_bbox_source"] = "synthetic_hxhy"
    path.write_text(json.dumps([converted]))
    with pytest.raises(ValueError, match="annotated human"):
        audited_records(tmp_path)


def test_visualization_selection_is_prediction_independent():
    records = [{"path": str(i)} for i in range(20)]
    assert visualization_indices(records) == visualization_indices(records)
    assert len(set(visualization_indices(records))) == 8
