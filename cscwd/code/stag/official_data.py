"""Read all official VAT train/test tracks and their frozen, read-only caches."""

import json
from pathlib import Path

from .cache import file_sha256, load_cache
from .feature_cache import load_feature_cache


def load_official_predictions(prediction_report, split):
    """Verify a completed single-frame cache without a development split file."""
    report = json.loads(Path(prediction_report).read_text())
    directory = Path(report["cache"]["path"])
    if not report["passed"] or not report["cache"]["passed"]:
        raise ValueError("Official predictions require a completed single-frame report")
    if file_sha256(directory / "index.json") != report["cache"]["index_sha256"]:
        raise ValueError("Official prediction index SHA256 mismatch")
    for item in report["cache"]["files"].values():
        if file_sha256(directory / item["path"]) != item["sha256"]:
            raise ValueError("Official prediction file SHA256 mismatch")
    index, arrays = load_cache(directory)
    tracks = index["tracks"]
    if any(not track["track_id"].startswith(split + "/") for track in tracks):
        raise ValueError(f"Prediction tracks do not belong to official {split}")
    count = 0
    for track in tracks:
        stop = count + len(track["frames"])
        if track["start"] != count or track["stop"] != stop:
            raise ValueError("Official prediction tracks are not a complete contiguous cache")
        count = stop
    if (len(tracks) != report["cache"]["tracks"] or count != index["person_frames"]
            or count != report["cache"]["person_frames"]
            or any(len(values) != count for values in arrays.values())):
        raise ValueError("Official prediction coverage differs from its report")
    return report, index, arrays


def _load_split(split, feature_cache, feature_report, prediction_report):
    prediction, index, arrays = load_official_predictions(prediction_report, split)
    feature_run = json.loads(Path(feature_report).read_text())
    feature_cache = Path(feature_cache)
    if not feature_run["passed"] or feature_run["mode"] != "cache":
        raise ValueError("Official training requires a completed frozen feature report")
    if (feature_cache.resolve() != Path(feature_run["cache"]["path"]).resolve()
            or file_sha256(feature_cache / "index.json") != feature_run["cache"]["index_sha256"]):
        raise ValueError("Official feature cache path or index SHA256 mismatch")
    feature_index, features = load_feature_cache(feature_cache)
    for item in feature_index["files"].values():
        print(f"Verifying frozen {split} feature file: {feature_cache / item['path']}", flush=True)
        if file_sha256(feature_cache / item["path"]) != item["sha256"]:
            raise ValueError("Official frozen feature file SHA256 mismatch")
    expected = {"prediction_index_sha256": prediction["cache"]["index_sha256"],
                "checkpoint_sha256": prediction["checkpoint"]["sha256"],
                "backbone_checkpoint_sha256": prediction["backbone_checkpoint"]["sha256"]}
    if (feature_index["sources"] != feature_run["sources"]
            or any(feature_index["sources"][key] != value for key, value in expected.items())):
        raise ValueError("Official frozen features and predictions have different sources")
    if split == "test" and (
            feature_run["official_split"] != "test"
            or feature_index["sources"]["prediction_report_sha256"] != file_sha256(prediction_report)):
        raise ValueError("Test features do not reference the matching official test predictions")
    image_paths = {frame["path"] for track in index["tracks"] for frame in track["frames"]}
    if (set(feature_index["image_paths"]) != image_paths
            or len(feature_index["image_paths"]) != len(image_paths)
            or len(features) != len(image_paths)
            or len(image_paths) != prediction["cache"]["unique_image_frames"]):
        raise ValueError("Official feature image coverage differs from the prediction cache")
    if feature_index["contract"] != feature_run["contract"]:
        raise ValueError("Official feature contract differs from its report")
    return {"prediction_index": index, "arrays": arrays, "feature_index": feature_index,
            "features": features, "tracks": index["tracks"]}, prediction, feature_run


def official_training_inputs(train_feature_cache, train_feature_report,
                             test_feature_cache, test_feature_report,
                             train_prediction_report, test_prediction_report):
    """Return every official train/test track; no validation manifest is consulted."""
    train, train_prediction, train_features = _load_split(
        "train", train_feature_cache, train_feature_report, train_prediction_report)
    test, test_prediction, test_features = _load_split(
        "test", test_feature_cache, test_feature_report, test_prediction_report)
    identity = {"model": train_prediction["model"],
                "checkpoint_sha256": train_prediction["checkpoint"]["sha256"],
                "backbone_checkpoint_sha256": train_prediction["backbone_checkpoint"]["sha256"],
                "prediction_protocol": train_prediction["protocol"]}
    other = {"model": test_prediction["model"],
             "checkpoint_sha256": test_prediction["checkpoint"]["sha256"],
             "backbone_checkpoint_sha256": test_prediction["backbone_checkpoint"]["sha256"],
             "prediction_protocol": test_prediction["protocol"]}
    if identity != other or train_features["contract"] != test_features["contract"]:
        raise ValueError("Official train/test model identity or feature contract differs")
    if set(train["feature_index"]["image_paths"]) & set(test["feature_index"]["image_paths"]):
        raise ValueError("Official train/test image paths overlap")
    if test_features["sources"]["train_feature_report_sha256"] != file_sha256(train_feature_report):
        raise ValueError("Test features reference a different train feature contract report")
    sources = {**identity,
        "train_prediction_report_sha256": file_sha256(train_prediction_report),
        "test_prediction_report_sha256": file_sha256(test_prediction_report),
        "train_feature_report_sha256": file_sha256(train_feature_report),
        "test_feature_report_sha256": file_sha256(test_feature_report),
        **{split: {
        "prediction_report_sha256": file_sha256(prediction_report),
        "feature_report_sha256": file_sha256(feature_report),
        "prediction_index_sha256": prediction["cache"]["index_sha256"],
        "feature_index_sha256": feature["cache"]["index_sha256"],
    } for split, prediction_report, feature_report, prediction, feature in (
        ("train", train_prediction_report, train_feature_report, train_prediction, train_features),
        ("test", test_prediction_report, test_feature_report, test_prediction, test_features))}}
    return {"train": train, "test": test, "sources": sources,
            "feature_contract": train_features["contract"]}
