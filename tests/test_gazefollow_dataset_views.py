import json

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from gazelle.dataloader import (
    GazeDataset,
    GazeFollowImageDataset,
    collate_gazefollow_images,
    collate_fn,
)
from scripts.train_coverage_router import unpack_batch


def _head(*, inout=1, gaze=(0.75, 0.25)):
    return {
        "bbox": [0, 0, 4, 4],
        "bbox_norm": [0.0, 0.0, 0.5, 0.5],
        "gazex": [gaze[0] * 8],
        "gazey": [gaze[1] * 8],
        "gazex_norm": [gaze[0]],
        "gazey_norm": [gaze[1]],
        "inout": inout,
    }


def _write_train_fixture(root):
    records = [
        {
            "path": "images/a.jpg",
            "heads": [_head(), _head(gaze=(0.25, 0.75))],
        },
        {
            "path": "images/b.jpg",
            "heads": [_head()],
        },
    ]
    image_dir = root / "images"
    image_dir.mkdir()
    for filename in ("a.jpg", "b.jpg"):
        Image.new("RGB", (8, 8), color="white").save(image_dir / filename)
    (root / "train_preprocessed.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    return records


def _tensor_transform(_image):
    return torch.zeros(3, 8, 8)


def test_image_indices_keep_every_head_from_an_image_group(tmp_path):
    _write_train_fixture(tmp_path)

    dataset = GazeDataset(
        "gazefollow",
        str(tmp_path),
        "train",
        _tensor_transform,
        image_indices=[0],
        augment=False,
        return_heatmap=False,
    )

    assert dataset.image_indices == (0,)
    assert dataset.data_idxs == [(0, 0), (0, 1)]
    assert len(dataset) == 2


def test_train_annotations_can_be_used_as_non_augmented_validation(tmp_path):
    _write_train_fixture(tmp_path)
    dataset = GazeDataset(
        "gazefollow",
        str(tmp_path),
        "train",
        _tensor_transform,
        image_indices=[1],
        augment=False,
        return_heatmap=False,
    )

    sample = dataset[0]

    assert dataset.aug is False
    assert dataset.return_heatmap is False
    assert len(sample) == 7
    assert sample[2] == [0.75]
    assert sample[3] == [0.25]

    batch = collate_fn([sample])
    model_input, *_ = unpack_batch(batch, torch.device("cpu"), training=False)
    assert model_input["_query_unit"] == "person"
    assert [len(image_bboxes) for image_bboxes in model_input["bboxes"]] == [1]


def test_image_grouped_evaluation_keeps_one_backbone_input_per_image(tmp_path):
    records = _write_train_fixture(tmp_path)
    dataset = GazeFollowImageDataset(
        str(tmp_path),
        "train",
        _tensor_transform,
        records=records,
    )

    batch = collate_gazefollow_images([dataset[0], dataset[1]])
    images, bboxes, gazex, gazey, inout, heights, widths = batch

    assert len(dataset) == 2
    assert dataset.person_count == 3
    assert images.shape == (2, 3, 8, 8)
    assert [len(image_bboxes) for image_bboxes in bboxes] == [2, 1]
    assert gazex == [[0.75], [0.25], [0.75]]
    assert gazey == [[0.25], [0.75], [0.25]]
    assert inout.tolist() == [1, 1, 1]
    assert heights == [8, 8, 8]
    assert widths == [8, 8, 8]

    model_input, *_ = unpack_batch(batch, torch.device("cpu"), training=False)
    assert model_input["_query_unit"] == "image"
    assert [len(image_bboxes) for image_bboxes in model_input["bboxes"]] == [2, 1]


def test_existing_train_dataset_defaults_remain_backward_compatible(tmp_path):
    _write_train_fixture(tmp_path)

    dataset = GazeDataset(
        "gazefollow",
        str(tmp_path),
        "train",
        _tensor_transform,
    )

    assert dataset.aug is True
    assert dataset.return_heatmap is True
    assert dataset.image_indices == (0, 1)


@pytest.mark.parametrize(
    ("indices", "error"),
    [
        ([0, 0], ValueError),
        ([-1], IndexError),
        ([2], IndexError),
    ],
)
def test_image_indices_are_validated(tmp_path, indices, error):
    _write_train_fixture(tmp_path)

    with pytest.raises(error):
        GazeDataset(
            "gazefollow",
            str(tmp_path),
            "train",
            _tensor_transform,
            image_indices=indices,
        )
