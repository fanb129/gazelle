import pytest

from gazelle.data_splits import grouped_holdout_split


def test_grouped_holdout_is_deterministic_and_has_no_group_leakage():
    groups = ["a.jpg", "a.jpg", "b.jpg", "c.jpg", "d.jpg", "d.jpg"]

    first = grouped_holdout_split(
        groups, validation_fraction=0.5, seed=3106
    )
    second = grouped_holdout_split(
        groups, validation_fraction=0.5, seed=3106
    )

    assert first == second
    train_groups = {groups[index] for index in first.train_indices}
    validation_groups = {groups[index] for index in first.validation_indices}
    assert train_groups.isdisjoint(validation_groups)
    assert sorted(first.train_indices + first.validation_indices) == list(
        range(len(groups))
    )


def test_grouped_holdout_metadata_records_reproducibility_fields():
    split = grouped_holdout_split(
        [f"image_{index}.jpg" for index in range(10)],
        validation_fraction=0.2,
        seed=42,
    )

    metadata = split.metadata()
    assert metadata["strategy"] == "stable_sha256_group_holdout_v1"
    assert metadata["seed"] == 42
    assert metadata["validation_fraction"] == 0.2
    assert metadata["train_group_count"] == 8
    assert metadata["validation_group_count"] == 2
    assert len(metadata["source_group_fingerprint"]) == 64
    assert len(metadata["train_group_fingerprint"]) == 64
    assert len(metadata["validation_group_fingerprint"]) == 64
    assert len(metadata["assignment_fingerprint"]) == 64


def test_assignment_fingerprint_changes_if_source_order_changes():
    first = grouped_holdout_split(
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        validation_fraction=0.25,
        seed=3106,
    )
    reordered = grouped_holdout_split(
        ["b.jpg", "a.jpg", "c.jpg", "d.jpg"],
        validation_fraction=0.25,
        seed=3106,
    )

    assert first.validation_group_fingerprint == reordered.validation_group_fingerprint
    assert first.source_group_fingerprint != reordered.source_group_fingerprint
    assert first.assignment_fingerprint != reordered.assignment_fingerprint


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_grouped_holdout_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="validation_fraction"):
        grouped_holdout_split(
            ["a.jpg", "b.jpg"],
            validation_fraction=fraction,
            seed=3106,
        )


def test_grouped_holdout_requires_two_distinct_groups():
    with pytest.raises(ValueError, match="distinct groups"):
        grouped_holdout_split(
            ["same.jpg", "same.jpg"],
            validation_fraction=0.5,
            seed=3106,
        )
