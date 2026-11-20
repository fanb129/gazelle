"""Deterministic, label-independent VAT train groups for temporal development.

The default group is the native annotation video directory. It conservatively
keeps its clips together; it is not a verified recording-level source ID.
An explicit mapping can merge annotation video directories from one source.
"""

from collections import defaultdict
from pathlib import PurePosixPath

from gazelle.data_splits import grouped_holdout_split


def _annotation_video(track):
    split, video, _clip, _person = PurePosixPath(track["track_id"]).parts
    if split != "train":
        raise ValueError(f"Development split accepts train tracks only: {track['track_id']}")
    return video


def _counts(tracks):
    return {
        "tracks": len(tracks),
        "person_frames": sum(len(track["frames"]) for track in tracks),
        "unique_image_frames": len(_image_paths(tracks)),
    }


def _image_paths(tracks):
    return {frame["path"] for track in tracks for frame in track["frames"]}


def build_development_split(tracks, validation_fraction=0.2, seed=20261117,
                            source_video_map=None):
    """Return a reproducible train/validation manifest without reading labels.

    ``tracks`` uses the prediction-cache index schema. Partition entries expose
    sorted ``track_ids``, ``group_ids``, ``annotation_videos``, and ``counts``.
    ``source_video_map`` may map any annotation video to a common source group;
    unmapped videos keep their annotation directory name as their group ID.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    tracks = sorted(tracks, key=lambda track: track["track_id"])
    track_ids = [track["track_id"] for track in tracks]
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("Duplicate track_id in development tracks")

    provided_mapping = dict(source_video_map or {})
    annotation_videos = sorted({_annotation_video(track) for track in tracks})
    unknown_videos = provided_mapping.keys() - set(annotation_videos)
    if unknown_videos:
        raise ValueError(f"Source mapping contains unknown annotation videos: {sorted(unknown_videos)}")
    effective_mapping = {video: provided_mapping.get(video, video) for video in annotation_videos}

    grouped_tracks = defaultdict(list)
    for track in tracks:
        grouped_tracks[effective_mapping[_annotation_video(track)]].append(track)
    group_ids = sorted(grouped_tracks)
    if len(group_ids) < 2:
        raise ValueError("Development split requires at least two source video groups")
    group_keys = [effective_mapping[_annotation_video(track)] for track in tracks]
    assignment = grouped_holdout_split(group_keys, validation_fraction=validation_fraction, seed=seed)
    validation_groups = {group_keys[index] for index in assignment.validation_indices}

    partition_tracks = {"train": [], "validation": []}
    groups = {}
    for group_id in group_ids:
        partition = "validation" if group_id in validation_groups else "train"
        members = grouped_tracks[group_id]
        partition_tracks[partition].extend(members)
        groups[group_id] = {
            "partition": partition,
            "annotation_videos": sorted({_annotation_video(track) for track in members}),
            "track_ids": [track["track_id"] for track in members],
            "counts": _counts(members),
        }

    partitions = {}
    for partition, members in partition_tracks.items():
        partitions[partition] = {
            "group_ids": sorted(group_id for group_id, group in groups.items()
                                if group["partition"] == partition),
            "annotation_videos": sorted({_annotation_video(track) for track in members}),
            "track_ids": sorted(track["track_id"] for track in members),
            "counts": _counts(members),
        }

    train_ids, validation_ids = (set(partitions[name]["track_ids"])
                                 for name in ("train", "validation"))
    train_images, validation_images = (_image_paths(partition_tracks[name])
                                       for name in ("train", "validation"))
    checks = {
        "source_video_groups_disjoint": not (
            set(partitions["train"]["group_ids"]) & set(partitions["validation"]["group_ids"])),
        "track_ids_disjoint": not (train_ids & validation_ids),
        "image_paths_disjoint": not (train_images & validation_images),
        "all_tracks_assigned": train_ids | validation_ids == set(track_ids),
        "person_frames_preserved": sum(partitions[name]["counts"]["person_frames"]
                                       for name in partitions) == _counts(tracks)["person_frames"],
        "all_images_assigned": train_images | validation_images == _image_paths(tracks),
    }
    if not all(checks.values()):
        raise ValueError(f"Development split verification failed: {checks}")
    return {
        "schema_version": 1,
        "protocol": "r1_development_split_v1",
        "official_input_split": "train",
        "seed": seed,
        "requested_validation_group_fraction": validation_fraction,
        "actual_validation_group_fraction": len(validation_groups) / len(group_ids),
        "grouping": {
            "annotation_video_field": "track_id: train/<annotation_video>/<clip>/<person>",
            "default_semantics": "conservative_annotation_video_directory_group; "
                                 "not_verified_independent_recording_id",
            "mapping_semantics": "optional_explicit_source_merge; unmapped_videos_keep_directory_name",
            "provided_source_video_map": dict(sorted(provided_mapping.items())),
            "annotation_video_to_group": effective_mapping,
            "selection": "stable_sha256_group_holdout_v1; "
                         "clamped_round_half_up_group_count; no_label_balancing",
            "assignment": assignment.metadata(),
        },
        "totals": {**_counts(tracks), "annotation_videos": len(annotation_videos),
                   "source_video_groups": len(group_ids)},
        "groups": groups,
        "partitions": partitions,
        "checks": checks,
        "passed": True,
    }
