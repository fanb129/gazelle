"""GOO-Real test queries: RGB and annotated human boxes, without augmentation."""

import json
from pathlib import Path
import pickle

import numpy as np
import torch

from data_prep.preprocess_gooreal import convert_record
from scripts.eval_gooreal import GoorealImageStore, validate_records


def audited_records(data_path):
    data_path = Path(data_path)
    records = json.loads((data_path / "gooreal_test_preprocessed.json").read_text())
    with (data_path / "testrealhumansNew.pickle").open("rb") as handle:
        raw = pickle.load(handle)
    validate_records(records)
    if len(records) != len(raw):
        raise ValueError("preprocessed and official test query counts differ")
    for index, (record, original) in enumerate(zip(records, raw)):
        expected = convert_record(original, index, "finalrealdatasetImgsV3")
        head = record["heads"][0]
        if len(record["heads"]) != 1 or head["inout"] != 1:
            raise ValueError("GOO-Real test requires one in-frame query per image")
        if expected.get("gooreal_warnings") or not head["gooreal_bbox_source"].startswith("label25_contains_hxhy:"):
            raise ValueError("GOO-Real requires annotated human boxes containing hx/hy")
        if record["path"] != expected["path"]:
            raise ValueError("preprocessed test image order differs from the official pickle")
        for field in ("bbox_norm", "gazex_norm", "gazey_norm"):
            np.testing.assert_allclose(head[field], expected["heads"][0][field], rtol=0, atol=1e-12)
    return records, {"images": len(records), "persons": len(records),
                     "annotated_human_boxes": len(records), "synthetic_or_detection_boxes": 0,
                     "annotation_coordinate_canvas": [640, 480],
                     "audit": "all paths, boxes and gaze points matched official raw test annotations"}


class GoorealTest:
    def __init__(self, records, data_path, transform):
        self.records = records
        self.store = GoorealImageStore(str(data_path))
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with self.store.open(record["path"]) as image:
            tensor = self.transform(image)
        return tensor, index


def collate(batch):
    images, indices = zip(*batch)
    return torch.stack(images), list(indices)
