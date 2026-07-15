from __future__ import annotations

import json
from pathlib import Path

import pytest

from AAAISelectiveGaze.data.split_builder import (
    SPLIT_NAMES,
    SplitRatios,
    build_splits,
)
from AAAISelectiveGaze.scripts.make_splits import main


def _gazefollow_frames(number_of_images: int = 20) -> list[dict]:
    frames = []
    for image_index in range(number_of_images):
        # Two records for a few images exercises grouping rather than merely
        # shuffling independent records.
        repeats = 2 if image_index % 5 == 0 else 1
        for repeat in range(repeats):
            frames.append(
                {
                    "path": f"images/{image_index:03d}.jpg",
                    "heads": [
                        {
                            "head_id": repeat,
                            "bbox_norm": [0.1, 0.1, 0.2, 0.2],
                            "inout": 1,
                        }
                    ],
                }
            )
    return frames


def _paths(records: list[dict]) -> set[str]:
    return {record["path"] for record in records}


def _head_sample_ids(records: list[dict]) -> set[str]:
    result = set()
    for record in records:
        frames = record.get("frames", [record])
        for frame in frames:
            result.update(head["sample_id"] for head in frame.get("heads", []))
    return result


def test_gazefollow_image_and_sample_ids_never_cross_splits() -> None:
    splits, manifest = build_splits(
        _gazefollow_frames(), dataset="gazefollow", seed=3106
    )

    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            assert _paths(splits[left]).isdisjoint(_paths(splits[right]))
            assert _head_sample_ids(splits[left]).isdisjoint(
                _head_sample_ids(splits[right])
            )
    assert manifest["leakage_check"] == {
        "group_intersections_empty": True,
        "sample_intersections_empty": True,
        "image_intersections_empty": True,
    }
    assert sum(part["num_groups"] for part in manifest["splits"].values()) == 20
    assert all(manifest["splits"][name]["num_groups"] > 0 for name in SPLIT_NAMES)


def test_split_is_deterministic_and_seed_controls_assignment() -> None:
    records = _gazefollow_frames(40)
    first_splits, first_manifest = build_splits(
        records, dataset="gazefollow", seed=123
    )
    second_splits, second_manifest = build_splits(
        records, dataset="gazefollow", seed=123
    )
    other_splits, _ = build_splits(records, dataset="gazefollow", seed=456)

    assert first_splits == second_splits
    assert first_manifest == second_manifest
    assert _paths(first_splits["base_train"]) != _paths(other_splits["base_train"])


def test_vat_sequence_list_is_split_as_complete_sequences() -> None:
    sequences = []
    for sequence_index in range(12):
        sequence_id = f"show/sequence_{sequence_index:02d}"
        sequences.append(
            {
                # Legacy VAT preprocessing calls the sequence identity path.
                "path": sequence_id,
                "frames": [
                    {
                        "path": f"{sequence_id}/{frame_index:04d}.jpg",
                        "heads": [{"head_id": 0, "inout": 1}],
                    }
                    for frame_index in range(3)
                ],
            }
        )

    splits, manifest = build_splits(
        sequences,
        dataset="vat",
        group_key="sequence_id",
        ratios=SplitRatios(0.5, 0.2, 0.2, 0.1),
        seed=7,
    )

    owners = {}
    for split_name, split_sequences in splits.items():
        for sequence in split_sequences:
            sequence_id = sequence["path"]
            assert sequence_id not in owners
            owners[sequence_id] = split_name
            assert len(sequence["frames"]) == 3  # builder did not flatten frames
            for frame in sequence["frames"]:
                assert frame["path"].startswith(sequence_id + "/")
                assert "sample_id" in frame["heads"][0]
    assert len(owners) == len(sequences)
    assert manifest["resolved_group_key"] == "path (sequence_id alias)"


def test_flat_vat_frames_require_a_sequence_identity() -> None:
    frames = [{"path": "seq/a/0001.jpg", "heads": []}]
    with pytest.raises(KeyError, match="Flat VAT input requires sequence_id"):
        build_splits(frames, dataset="vat")


def test_vat_rejects_an_image_claimed_by_two_sequences() -> None:
    sequences = [
        {
            "sequence_id": sequence_id,
            "frames": [{"path": "shared/0001.jpg", "heads": []}],
        }
        for sequence_id in ("sequence-a", "sequence-b")
    ]
    with pytest.raises(ValueError, match="image path.*multiple groups"):
        build_splits(sequences, dataset="vat")


@pytest.mark.parametrize(
    "ratios",
    [
        SplitRatios(0.8, 0.1, 0.1, 0.1),
        SplitRatios(0.9, 0.1, 0.0, -0.0 - 0.01),
    ],
)
def test_ratios_must_be_non_negative_and_sum_to_one(ratios: SplitRatios) -> None:
    with pytest.raises(ValueError, match="Split ratios"):
        build_splits(_gazefollow_frames(4), dataset="gazefollow", ratios=ratios)


def test_cli_writes_expected_files_and_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "gazefollow.json"
    output_path = tmp_path / "splits"
    input_path.write_text(json.dumps(_gazefollow_frames()), encoding="utf-8")

    exit_code = main(
        [
            "--dataset",
            "gazefollow",
            "--input-json",
            str(input_path),
            "--output-dir",
            str(output_path),
            "--seed",
            "19",
        ]
    )

    assert exit_code == 0
    assert {path.name for path in output_path.iterdir()} == {
        "base_train.json",
        "base_val.json",
        "risk_train.json",
        "risk_calibration.json",
        "manifest.json",
    }
    manifest = json.loads((output_path / "manifest.json").read_text())
    assert manifest["seed"] == 19
    assert manifest["input"]["path"] == str(input_path)
    assert len(manifest["input"]["sha256"]) == 64
