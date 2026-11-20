"""Synthetic raw-prediction cache -> portable train/test human review package."""

import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from cscwd.code.stag.review_package import build_review_package, sha256


def make_prediction_cache(root, split):
    data_root, cache_root = root / "images_root", root / (split + "_predictions")
    cache_root.mkdir(parents=True)
    tracks, heatmaps, scores, points = [], [], [], []
    for person in range(6):
        frames = []
        start = len(scores)
        for offset, number in enumerate((1, 2, 3, 4, 8, 9)):
            relative = f"images/{split}/video_{person}/clip/{number:08d}.jpg"
            image_path = data_root / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (112, 64), (40 + person * 20, 70 + offset * 15, 110)).save(image_path)
            # A mix of TP/TN/FN/FP, with varied localization error.
            inside = int(offset % 3 != 2)
            score = (0.9, 0.2, 0.9, 0.8, 0.1, 0.2)[offset]
            x, y = (person + offset * 7) % 64, (person * 7 + offset) % 64
            heatmap = np.zeros((64, 64), dtype=np.float32)
            heatmap[y, x] = 0.75
            heatmaps.append(heatmap)
            points.append((x / 64, y / 64))
            scores.append(score)
            frames.append({
                "path": relative, "frame_index": number, "person_id": str(person),
                "bbox_norm": [0.1, 0.1, 0.3, 0.5],
                "gazex_norm": [0.5 if inside else -1.0],
                "gazey_norm": [0.5 if inside else -1.0], "inout": inside,
            })
        tracks.append({"track_id": f"{split}/video_{person}/clip/{person}",
                       "person_id": str(person), "source_csv": "synthetic.txt",
                       "sequence_path": f"images/{split}/video_{person}/clip",
                       "width": 112, "height": 64, "start": start, "stop": len(scores),
                       "frames": frames})
    arrays = {"heatmap": np.stack(heatmaps), "inout": np.asarray(scores, dtype=np.float32),
              "argmax_xy": np.asarray(points, dtype=np.float32)}
    files = {}
    for name, array in arrays.items():
        path = cache_root / (name + ".npy")
        np.save(path, array)
        files[name] = {"path": path.name, "shape": list(array.shape), "dtype": "float32",
                       "sha256": sha256(path)}
    metadata = {
        "checkpoint": {"path": "synthetic_base.pt", "sha256": "synthetic-base"},
        "backbone_checkpoint": {"path": "synthetic_backbone.pth", "sha256": "synthetic-backbone"},
        "track_json": {"path": split + "_tracks.json", "sha256": "synthetic-" + split},
        "protocol": {"boxes": "T1_current_frame_raw_ground_truth", "labels_used_in_forward": False,
                     "temporal_context": "none"},
    }
    index = {"schema_version": 1, "task": "T2", "metadata": metadata,
             "person_frames": len(scores), "track_count": len(tracks), "tracks": tracks,
             "files": files, "checks": {"all_person_frames_cached": True}, "passed": True}
    (cache_root / "index.json").write_text(json.dumps(index))
    return data_root, cache_root


def test_train_test_review_package_roundtrip(tmp_path):
    data_root, train_cache = make_prediction_cache(tmp_path, "train")
    _, test_cache = make_prediction_cache(tmp_path, "test")
    output, reports = tmp_path / "review", tmp_path / "reports"
    provenance = tmp_path / "provenance.md"
    report = build_review_package(
        data_root, train_cache, test_cache, output, reports, provenance,
        counts={"large_l2": 2, "inout_false_negative": 2, "inout_false_positive": 2,
                "medium_l2": 2, "correct_control": 2},
        history_frames=1, max_history_frames=2, max_per_track=2, max_per_video=2,
        image_max_side=80, archive=True, command="synthetic_review_check_only",
    )
    manifest = json.loads((output / "case_manifest.json").read_text())
    assert {case["split"] for case in manifest["cases"]} == {"train", "test"}
    assert report["protocol"]["test_used_for_manual_analysis"]
    assert report["protocol"]["test_used_for_method_development"]
    assert report["protocol"]["validation_split"] is None
    assert report["protocol"]["test_is_independent_validation"] is False
    assert "__REVIEW_DATA__" not in (output / "index.html").read_text()
    assert report["case_count"] == len(manifest["cases"])
    assert report["case_manifest_sha256"] == sha256(output / "case_manifest.json")
    assert report["index_html_sha256"] == sha256(output / "index.html")
    assert report["archive"]["sha256"] == sha256(report["archive"]["path"])
    assert report["case_manifest_sha256"] in provenance.read_text()
    for case in manifest["cases"]:
        numbers = [frame["frame_index"] for frame in case["frames"]]
        assert numbers[-1] == case["anchor_frame_index"]
        assert all(current == previous + 1 for previous, current in zip(numbers, numbers[1:]))
        for frame in case["frames"]:
            assert frame["relative_to_anchor"] <= 0
            with Image.open(output / frame["image"]) as image:
                assert max(image.size) <= 80
                assert abs(image.width / image.height - 112 / 64) < 0.04
            with Image.open(output / frame["heatmap"]) as heatmap:
                assert heatmap.size == (64, 64)
                assert np.asarray(heatmap).max() == round(0.75 * 255)
            with Image.open(output / frame["head_crop"]) as head:
                assert 0 < min(head.size) <= max(head.size) <= 192
    labels = json.loads((output / "annotations_blank.json").read_text())
    assert labels["package_id"] == report["package_id"]
    assert {label["case_id"] for label in labels["labels"]} == {case["case_id"] for case in manifest["cases"]}
    assert labels["artifact_type"] == "cscwd_failure_review_labels"
    assert all(not label["current_guess"] and label["gt_revealed_at"] == "" for label in labels["labels"])
    with zipfile.ZipFile(report["archive"]["path"]) as archive:
        assert "index.html" in archive.namelist()
        assert any(name.startswith("assets/rgb/") for name in archive.namelist())
    selected = json.loads((reports / "selected_cases.json").read_text())
    assert all("frames" not in case for case in selected["cases"])
    with pytest.raises(FileExistsError):
        build_review_package(data_root, train_cache, test_cache, output, reports, provenance)


def test_cache_sha_mismatch_is_rejected(tmp_path):
    data_root, train_cache = make_prediction_cache(tmp_path, "train")
    _, test_cache = make_prediction_cache(tmp_path, "test")
    path = train_cache / "inout.npy"
    array = np.load(path)
    array[0] = 0.4
    np.save(path, array)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_review_package(data_root, train_cache, test_cache,
                             tmp_path / "review", tmp_path / "reports", tmp_path / "provenance.md")
