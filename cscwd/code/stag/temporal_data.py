"""Ordered person-track chunks with persistent batch slots for causal TBPTT.

Only metadata determines sequence boundaries. Ground-truth gaze and state are
used to construct the current frame's loss targets, never the model inputs or
the memory-reset mask. Frozen scene features are shared by image path.
"""

import hashlib

import numpy as np
import torch

from gazelle.utils import get_heatmap


def contiguous_segments(tracks, seed=None):
    """Return chronological contiguous segments in a reproducible epoch order.

    Segments separated by a frame gap have independent memory. A seeded order
    may shuffle those segments, but never reorders frames within a segment.
    Each item is ``(track, start_position, stop_position)`` in local frame units.
    """
    segments = []
    for track in sorted(tracks, key=lambda item: item["track_id"]):
        frames = track["frames"]
        boundaries = [0, *(index for index in range(1, len(frames))
                           if frames[index]["frame_index"] - frames[index - 1]["frame_index"] != 1),
                      len(frames)]
        segments.extend((track, first, stop) for first, stop in zip(boundaries, boundaries[1:]))
    if seed is not None:
        segments.sort(key=lambda item: (
            hashlib.sha256(f"{seed}\0{item[0]['track_id']}\0{item[1]}".encode()).hexdigest(),
            item[0]["track_id"], item[1]))
    return segments


def _empty_batch(batch_size, clip_frames, features):
    feature_dtype = torch.from_numpy(np.empty((), dtype=features.dtype)).dtype
    return {
        "features": torch.zeros((batch_size, clip_frames, *features.shape[1:]), dtype=feature_dtype),
        "raw_heatmap": torch.zeros((batch_size, clip_frames, 64, 64), dtype=torch.float32),
        "raw_inout": torch.zeros((batch_size, clip_frames), dtype=torch.float32),
        "bboxes": torch.zeros((batch_size, clip_frames, 4), dtype=torch.float32),
        "target_heatmap": torch.zeros((batch_size, clip_frames, 64, 64), dtype=torch.float32),
        "target_inout": torch.zeros((batch_size, clip_frames), dtype=torch.float32),
        "valid": torch.zeros((batch_size, clip_frames), dtype=torch.bool),
        "reset": torch.zeros((batch_size, clip_frames), dtype=torch.bool),
        "global_indices": torch.full((batch_size, clip_frames), -1, dtype=torch.long),
        "frame_indices": torch.full((batch_size, clip_frames), -1, dtype=torch.long),
        "track_ids": [None] * batch_size,
    }


def iter_track_batches(tracks, prediction_arrays, feature_index, features,
                       clip_frames=16, batch_size=8, seed=None):
    """Yield CPU tensors while preserving each slot's state across chunks.

    A slot holds one contiguous segment. If the segment ends before the chunk
    ends, its remaining positions are padding; another segment starts only in
    the next batch, with ``reset[slot, 0]`` set. Continuing slots do not reset.
    The caller carries model state across yielded batches and only detaches its
    gradient graph for TBPTT. Validation uses the same iterator without a seed
    and carries state throughout each contiguous track segment.

    Prediction arrays keep their full-cache global indices. ``global_indices``
    maps valid outputs back to that cache and uses -1 for padding. All tensor
    buffers own their values, so read-only mmap inputs stay unchanged.
    """
    if clip_frames < 1 or batch_size < 1:
        raise ValueError("clip_frames and batch_size must be positive")
    image_to_index = {path: index for index, path in enumerate(feature_index["image_paths"])}
    segments = iter(contiguous_segments(tracks, seed))
    slots = [None] * batch_size
    while True:
        for slot in range(batch_size):
            if slots[slot] is None:
                segment = next(segments, None)
                if segment is not None:
                    track, first, stop = segment
                    slots[slot] = (track, first, stop, True)
        if all(segment is None for segment in slots):
            return
        batch = _empty_batch(batch_size, clip_frames, features)
        for slot, segment in enumerate(slots):
            if segment is None:
                continue
            track, first, segment_stop, starts_segment = segment
            stop = min(first + clip_frames, segment_stop)
            frames = track["frames"][first:stop]
            length = len(frames)
            global_first, global_stop = track["start"] + first, track["start"] + stop
            image_indices = [image_to_index[frame["path"]] for frame in frames]
            batch["features"][slot, :length] = torch.from_numpy(features[image_indices])
            batch["raw_heatmap"][slot, :length] = torch.from_numpy(
                np.array(prediction_arrays["heatmap"][global_first:global_stop], copy=True))
            batch["raw_inout"][slot, :length] = torch.from_numpy(
                np.array(prediction_arrays["inout"][global_first:global_stop], copy=True))
            batch["bboxes"][slot, :length] = torch.tensor([frame["bbox_norm"] for frame in frames])
            batch["target_inout"][slot, :length] = torch.tensor([frame["inout"] for frame in frames])
            for position, frame in enumerate(frames):
                if frame["inout"] == 1:
                    batch["target_heatmap"][slot, position] = get_heatmap(
                        frame["gazex_norm"][0], frame["gazey_norm"][0], 64, 64)
            batch["valid"][slot, :length] = True
            batch["reset"][slot, 0] = starts_segment
            batch["global_indices"][slot, :length] = torch.arange(global_first, global_stop)
            batch["frame_indices"][slot, :length] = torch.tensor([frame["frame_index"] for frame in frames])
            batch["track_ids"][slot] = track["track_id"]
            slots[slot] = None if stop == segment_stop else (track, stop, segment_stop, False)
        yield batch
