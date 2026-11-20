"""Synthetic checks for raw-cache contracts and reproducible review selection."""

from collections import Counter
import json

import numpy as np
import pytest

from cscwd.code.stag.failure_cases import PredictionCache, select_cases


def write_cache(root, specs, split="train"):
    root.mkdir(parents=True, exist_ok=True)
    count = sum(len(frames) for _, frames in specs)
    heatmap = np.zeros((count, 64, 64), dtype=np.float32)
    heatmap[:, 10, 10] = 1
    points = np.full((count, 2), 10 / 64, dtype=np.float32)
    scores = np.empty(count, dtype=np.float32)
    tracks = []
    row = 0
    for name, annotations in specs:
        video, clip, person = name.split("/")
        frames = []
        start = row
        for number, error, gt, score in annotations:
            frames.append({"frame_index": number, "path": f"images/{video}/{clip}/{number:08d}.jpg",
                           "person_id": person, "bbox_norm": [0.1, 0.1, 0.3, 0.4],
                           "gazex_norm": [10 / 64 + error if gt else -1],
                           "gazey_norm": [10 / 64 if gt else -1], "inout": gt})
            scores[row] = score
            row += 1
        tracks.append({"track_id": f"{split}/{name}", "person_id": person,
                       "sequence_path": f"images/{video}/{clip}", "width": 320, "height": 240,
                       "start": start, "stop": row, "frames": frames})
    files = {}
    for name, array in (("heatmap", heatmap), ("inout", scores), ("argmax_xy", points)):
        np.save(root / f"{name}.npy", array)
        files[name] = {"path": f"{name}.npy", "shape": list(array.shape), "dtype": "float32"}
    index = {"task": "T2", "passed": True, "person_frames": count, "track_count": len(tracks),
             "files": files, "tracks": tracks,
             "metadata": {"protocol": {"temporal_context": "none", "labels_used_in_forward": False}}}
    (root / "index.json").write_text(json.dumps(index))
    return root


def varied_specs(videos=8, persons=3):
    return [(f"video_{video}/clip/person_{person}", [
        (1, 0.00, 1, 0.8), (2, 0.02, 1, 0.8), (3, 0.15, 1, 0.7),
        (4, 0.25, 1, 0.4), (5, 0.35, 1, 0.9), (6, 0.60, 1, 0.2),
        (7, 0.00, 0, 0.8), (8, 0.00, 0, 0.3),
    ]) for video in range(videos) for person in range(persons)]


def test_original_row_order_and_offline_scores(tmp_path):
    specs = [("z_video/clip/p", [(1, 0, 1, 0.5), (3, 0.2, 1, 0.49)]),
             ("a_video/clip/p", [(1, 0, 0, 0.5), (2, 0, 0, 0.49)])]
    cache = PredictionCache(write_cache(tmp_path / "cache", specs), "train")
    assert [frame["track_id"] for frame in cache.frames] == ["train/z_video/clip/p"] * 2 + ["train/a_video/clip/p"] * 2
    assert [frame["row"] for frame in cache.frames] == list(range(4))
    assert [frame["local_index"] for frame in cache.frames] == [0, 1, 0, 1]
    np.testing.assert_array_equal(cache.segment_start, [0, 1, 2, 2])
    assert cache.frames[1]["l2"] == pytest.approx(0.2)
    assert cache.frames[2]["l2"] is None
    assert not cache.heatmap.flags.writeable and not cache.inout.flags.writeable
    result = select_cases(cache, counts={"correct_control": 0})
    assert result["summary"]["confusion"] == {"TP": 1, "TN": 1, "FP": 1, "FN": 1}
    assert result["summary"]["l2_mean"] == pytest.approx(0.1)
    json.dumps(result, allow_nan=False)


def test_reproducible_strata_global_caps_and_control_mix(tmp_path):
    cache = PredictionCache(write_cache(tmp_path / "cache", varied_specs()), "train")
    counts = {"large_l2": 4, "inout_false_negative": 4, "inout_false_positive": 4,
              "medium_l2": 4, "correct_control": 6}
    kwargs = {"counts": counts, "max_per_track": 1, "max_per_video": 3}
    first = select_cases(cache, seed=0, **kwargs)
    repeated = select_cases(cache, seed=0, **kwargs)
    assert first == repeated
    assert first["cases"] != select_cases(cache, seed=1, **kwargs)["cases"]
    track_counts = Counter(case["track_id"] for case in first["cases"])
    video_counts = Counter(case["video_id"] for case in first["cases"])
    assert max(track_counts.values()) <= 1 and max(video_counts.values()) <= 3
    assert len({case["anchor_row"] for case in first["cases"]}) == len(first["cases"])
    for case in first["cases"]:
        category, anchor = case["category"], cache.frames[case["anchor_row"]]
        if category == "large_l2":
            assert anchor["inout"] == 1 and anchor["l2"] >= first["summary"]["large_l2_threshold"]
        elif category == "inout_false_negative":
            assert anchor["inout"] == 1 and anchor["score"] < 0.5
        elif category == "inout_false_positive":
            assert anchor["inout"] == 0 and anchor["score"] >= 0.5
        elif category == "medium_l2":
            q = first["summary"]["l2_quantiles"]
            assert q["p40"] <= anchor["l2"] <= q["p60"]
        else:
            assert (anchor["score"] >= 0.5) == bool(anchor["inout"])
    assert first["summary"]["correct_control_in_frame"] > 0
    assert first["summary"]["correct_control_out_of_frame"] > 0
    assert first["summary"]["requested_total"] == first["summary"]["selected_total"] + first["summary"]["shortfall_total"]
    json.dumps(first, allow_nan=False)


def test_history_stops_at_gaps_and_windows_do_not_overlap(tmp_path):
    numbers = list(range(1, 13)) + list(range(30, 61))
    specs = [("video/clip/person", [(number, 0.4, 1, 0.8) for number in numbers])]
    cache = PredictionCache(write_cache(tmp_path / "cache", specs), "train")
    result = select_cases(cache, counts={"large_l2": 8}, max_per_track=8, max_per_video=8,
                          history_frames=4, max_history_frames=8)
    assert result["summary"]["categories"]["large_l2"]["candidate_count"] == len(numbers)
    assert result["summary"]["categories"]["large_l2"]["shortfall"] > 0
    assert result["summary"]["categories"]["large_l2"]["skipped_by_constraint"]["overlapping_history_window"] > 0
    windows = []
    for case in result["cases"]:
        frames = case["frames"]
        assert 1 <= len(frames) <= 9
        assert frames[-1]["row"] == case["anchor_row"]
        assert all(right["frame_index"] == left["frame_index"] + 1 for left, right in zip(frames, frames[1:]))
        assert all(frame["track_id"] == case["track_id"] for frame in frames)
        assert case["default_history"] <= 4
        assert case["included_history"] == len(frames) - 1
        if case["anchor_frame_index"] >= 30:
            assert frames[0]["frame_index"] >= 30
        windows.append((frames[0]["row"], frames[-1]["row"]))
    assert all(last < next_first for (_, last), (next_first, _) in zip(sorted(windows), sorted(windows)[1:]))


def test_sampling_limits_are_not_silently_relaxed(tmp_path):
    cache = PredictionCache(write_cache(tmp_path / "cache", varied_specs(videos=1, persons=5)), "train")
    result = select_cases(cache, counts={"large_l2": 20, "inout_false_positive": 20},
                          max_per_track=1, max_per_video=2)
    assert len(result["cases"]) == 2
    assert result["summary"]["shortfall_total"] == 38
    assert sum(values["skipped_by_constraint"].get("max_per_video", 0)
               for values in result["summary"]["categories"].values()) > 0


@pytest.mark.parametrize("fault", ["split", "nonfinite", "coordinates", "row_gap", "future_order"])
def test_cache_contract_rejects_invalid_inputs(tmp_path, fault):
    root = write_cache(tmp_path / "cache", varied_specs(videos=1, persons=1))
    if fault == "split":
        with pytest.raises(ValueError, match="split"):
            PredictionCache(root, "test")
        return
    if fault in ("nonfinite", "coordinates"):
        filename = "inout.npy" if fault == "nonfinite" else "argmax_xy.npy"
        array = np.load(root / filename, mmap_mode="r+")
        array[0] = np.nan if fault == "nonfinite" else [0.5, 0.5]
        array.flush()
    else:
        index = json.loads((root / "index.json").read_text())
        if fault == "row_gap":
            index["tracks"][0]["start"] = 1
        else:
            index["tracks"][0]["frames"][1]["frame_index"] = 0
        (root / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        PredictionCache(root, "train")
