import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.analyze_feature_oversmoothing import (
    compute_crowd_token_cosine,
    compute_effective_rank,
    compute_foreground_background_contrast,
    compute_inter_person_boundary_separability,
    iter_vat_crowd_samples,
    masks_from_boxes,
)


def test_crowd_token_cosine_uses_only_crowded_head_tokens():
    features = np.ones((3, 4, 4), dtype=np.float32)
    masks = [np.eye(4, dtype=bool), np.fliplr(np.eye(4, dtype=bool))]

    result = compute_crowd_token_cosine(features, masks)

    assert result["computed"] is True
    assert result["sample_count"] == 1
    assert result["skipped_count"] == 0
    assert result["value"] == 1.0


def test_crowd_token_cosine_records_skip_when_region_is_too_small():
    features = np.ones((2, 2, 2), dtype=np.float32)
    masks = [np.array([[True, False], [False, False]])]

    result = compute_crowd_token_cosine(features, masks)

    assert result["computed"] is False
    assert result["sample_count"] == 0
    assert result["skipped_count"] == 1
    assert result["skip_reasons"] == {"crowd_region_lt_two_tokens": 1}


def test_boundary_separability_requires_multiple_boxes_and_boundary_tokens():
    features = np.zeros((2, 5, 5), dtype=np.float32)
    features[0, :, :2] = 1.0
    features[1, :, 3:] = 1.0
    masks = masks_from_boxes([[0.0, 0.0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0]], 5, 5)

    result = compute_inter_person_boundary_separability(features, masks)

    assert result["computed"] is True
    assert result["sample_count"] == 1
    assert result["value"] > 0.0


def test_boundary_separability_records_skip_for_single_person():
    features = np.ones((2, 4, 4), dtype=np.float32)
    masks = masks_from_boxes([[0.0, 0.0, 0.5, 0.5]], 4, 4)

    result = compute_inter_person_boundary_separability(features, masks)

    assert result["computed"] is False
    assert result["skip_reasons"] == {"requires_at_least_two_regions": 1}


def test_foreground_background_contrast_compares_valid_token_sets():
    features = np.zeros((2, 4, 4), dtype=np.float32)
    features[0, :2, :2] = 1.0
    features[1, 2:, 2:] = 1.0
    masks = masks_from_boxes([[0.0, 0.0, 0.5, 0.5]], 4, 4)

    result = compute_foreground_background_contrast(features, masks)

    assert result["computed"] is True
    assert result["sample_count"] == 1
    assert result["skipped_count"] == 0
    assert result["value"] > 0.0


def test_effective_rank_reports_optional_metric():
    features = np.zeros((3, 3, 3), dtype=np.float32)
    features[0] = np.arange(9, dtype=np.float32).reshape(3, 3)
    features[1] = 1.0
    features[2] = np.eye(3, dtype=np.float32)

    result = compute_effective_rank(features)

    assert result["computed"] is True
    assert result["optional"] is True
    assert result["value"] >= 1.0


def test_synthetic_smoke_writes_backbone_layer_metric_rows(tmp_path):
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_feature_oversmoothing.py"
    output = tmp_path / "feature_oversmoothing_synthetic.json"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic_smoke",
            "--backbones",
            "dinov2_vitb16",
            "dinov3_vitb16",
            "--layers",
            "shallow",
            "last",
            "--metrics",
            "crowd_token_cosine",
            "inter_person_boundary_separability",
            "foreground_background_contrast",
            "effective_rank",
            "layerwise_probe",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output.read_text())

    assert payload["synthetic_smoke"] is True
    assert payload["sample_count"] == 1
    assert len(payload["rows"]) == 4
    first = payload["rows"][0]
    assert first["backbone"] == "dinov2_vitb16"
    assert first["resolved_backbone"] == "dinov2_vitb14"
    assert first["layer"] == "shallow"
    assert first["metrics"]["crowd_token_cosine"]["computed"] is True
    assert first["metrics"]["layerwise_probe"]["computed"] is False
    assert first["metrics"]["layerwise_probe"]["optional"] is True


def test_cli_rejects_unknown_metric(tmp_path):
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_feature_oversmoothing.py"
    output = tmp_path / "feature_oversmoothing_invalid.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic_smoke",
            "--metrics",
            "not_a_metric",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Unsupported metrics: not_a_metric" in result.stderr


def test_vat_crowd_samples_resolve_frame_path_relative_to_data_root(tmp_path):
    crowd_json = tmp_path / "test_crowd_ge4.json"
    crowd_json.write_text(
        json.dumps(
            [
                {
                    "path": "images/test/seq001",
                    "frames": [
                        {
                            "path": "images/test/seq001/000001.jpg",
                            "heads": [{"bbox_norm": [0.1, 0.1, 0.2, 0.2]}],
                        }
                    ],
                }
            ]
        )
    )

    samples = list(iter_vat_crowd_samples("/vat/root", str(crowd_json), max_samples=1))

    assert samples[0]["image_path"] == "/vat/root/images/test/seq001/000001.jpg"
