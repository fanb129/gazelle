"""Actual selector-bank streaming, identities and causal observation isolation."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT / "cscwd/code"), str(REPO_ROOT / "cscwd/code/scripts")]

from test_official_data import make_official_fixture
from stag.cache import file_sha256, write_json
from stag.official_data import _load_split
from stag.selector_bank import FILES, VERSION, build_selector_bank, load_selector_bank


class SelectorBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        fixture = make_official_fixture(cls.root / "sources")
        cls.part, prediction, _ = _load_split("train", fixture["train_feature_cache"],
            fixture["train_feature_report"], fixture["train_prediction_report"])
        cls.sources = {"checkpoint_sha256": prediction["checkpoint"]["sha256"],
                       "prediction_index_sha256": prediction["cache"]["index_sha256"]}
        cls.metadata = {"split": "train", "version": VERSION, "plan_sha256": "a" * 64}
        cls.original_sources = {path: file_sha256(path) for path in (cls.root / "sources").rglob("*")
                                if path.is_file()}
        cls.record = build_selector_bank(cls.part, cls.root / "bank", cls.sources,
                                         cls.metadata, device="cpu")
        cls.bank = load_selector_bank(cls.record)

    def build(self, name, part=None):
        return build_selector_bank(part or self.part, self.root / name,
                                   self.sources, self.metadata, device="cpu")

    def test_complete_read_only_bank_matches_global_raw_rows_and_sources(self):
        record, bank = self.record, self.bank
        self.assertTrue(record["passed"])
        self.assertEqual(record["coverage"], {"tracks": 2, "person_frames": 24, "unique_image_frames": 12})
        self.assertEqual(record["index_sha256"], file_sha256(Path(record["path"]) / "index.json"))
        self.assertEqual(bank["index"]["tracks"], self.part["tracks"])
        self.assertEqual(bank["index"]["sources"], self.sources)
        self.assertEqual(bank["index"]["metadata"], self.metadata)
        self.assertEqual(bank["index"]["feature_channels"], 4)
        self.assertEqual(bank["index"]["candidate_names"],
            ["current", "current_grid", "ema_075", "grid_ema_075", "transport_previous"])
        self.assertEqual(bank["heatmaps"].shape, (24, 5, 64, 64))
        self.assertEqual(bank["observations"].shape, (24, 41))
        np.testing.assert_array_equal(bank["heatmaps"][:, 0], self.part["arrays"]["heatmap"])
        np.testing.assert_array_equal(bank["inout"], self.part["arrays"]["inout"])
        np.testing.assert_array_equal(bank["candidate_xy"][:, 0], self.part["arrays"]["argmax_xy"])
        for name in FILES:
            self.assertFalse(bank[name].flags.writeable)
            self.assertEqual(record["files"][name]["sha256"],
                             file_sha256(Path(record["path"]) / FILES[name]))
        self.assertTrue(all(bank["index"]["checks"].values()))
        self.assertEqual(self.original_sources,
                         {path: file_sha256(path) for path in self.original_sources})

    def test_gt_changes_index_labels_but_no_generated_array(self):
        changed = {**self.part, "tracks": copy.deepcopy(self.part["tracks"])}
        for track in changed["tracks"]:
            for frame in track["frames"]:
                frame["inout"] = 1 - frame["inout"]
                frame["gazex_norm"], frame["gazey_norm"] = [.98], [.03]
        record = self.build("changed_gt", changed)
        for name in FILES:
            self.assertEqual(record["files"][name]["values_sha256"],
                             self.record["files"][name]["values_sha256"])
        self.assertNotEqual(record["index_sha256"], self.record["index_sha256"])

    def test_future_observations_do_not_change_earlier_rows(self):
        changed = {**self.part, "arrays": {**self.part["arrays"],
            "heatmap": np.array(self.part["arrays"]["heatmap"]),
            "argmax_xy": np.array(self.part["arrays"]["argmax_xy"]),
            "inout": np.array(self.part["arrays"]["inout"])},
            "features": np.array(self.part["features"])}
        earlier, future_paths = [], set()
        for track in self.part["tracks"]:
            for offset, frame in enumerate(track["frames"]):
                position = track["start"] + offset
                if frame["frame_index"] >= 7:
                    changed["arrays"]["heatmap"][position] = .01
                    changed["arrays"]["heatmap"][position, 3, 60] = .99
                    changed["arrays"]["argmax_xy"][position] = [60 / 64, 3 / 64]
                    changed["arrays"]["inout"][position] = .67
                    future_paths.add(frame["path"])
                else:
                    earlier.append(position)
        for row, path in enumerate(self.part["feature_index"]["image_paths"]):
            if path in future_paths:
                changed["features"][row] *= -1
        altered = load_selector_bank(self.build("changed_future", changed))
        for name in FILES:
            np.testing.assert_array_equal(altered[name][earlier], self.bank[name][earlier])

    def test_tracks_and_gaps_reset_history_and_both_ema_states(self):
        changed = {**self.part, "tracks": copy.deepcopy(self.part["tracks"])}
        for frame in changed["tracks"][0]["frames"][5:]:
            frame["frame_index"] += 2
        bank = load_selector_bank(self.build("gap", changed))
        self.assertEqual(np.flatnonzero(~bank["history_available"]).tolist(), [0, 5, 12])
        for row in (0, 5, 12):
            for column in (2, 4):
                np.testing.assert_array_equal(bank["heatmaps"][row, column], bank["heatmaps"][row, 0])
            np.testing.assert_array_equal(bank["heatmaps"][row, 3], bank["heatmaps"][row, 1])

    def test_repeated_output_is_rejected_and_original_bank_is_unchanged(self):
        with self.assertRaises(FileExistsError):
            self.build("bank")
        load_selector_bank(self.record)

    def test_tampered_values_are_rejected_by_file_hash(self):
        record = self.build("tampered")
        path = Path(record["path"]) / FILES["observations"]
        values = np.load(path, mmap_mode="r+")
        values[0, 0] += 1
        values.flush()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_selector_bank(record)

    def test_header_is_checked_without_hashing_and_index_tamper_is_rejected(self):
        record = self.build("wrong_header")
        path = Path(record["path"]) / FILES["observations"]
        np.save(path, np.zeros((24, 40), dtype=np.float32), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "header"):
            load_selector_bank(record, verify=False)
        record = self.build("wrong_index")
        index_path = Path(record["path"]) / "index.json"
        index = load_selector_bank(record)["index"]
        index["metadata"]["plan_sha256"] = "b" * 64
        write_json(index_path, index)
        with self.assertRaisesRegex(ValueError, "index SHA256"):
            load_selector_bank(record)


if __name__ == "__main__":
    unittest.main()
