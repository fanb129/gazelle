"""Chronological, stateful chunks from frozen per-person task features."""

import random

import numpy as np
import torch

from gazelle.utils import get_heatmap


def chunk_rows(tracks, batch_size, clip_frames, shuffle=False, seed=0):
    """Keep a track in its batch slot until exhausted; never shuffle its frames.

    A new iterator is an epoch/rollout boundary. Within it, reset depends only
    on person identity and original frame numbers, including across chunks.
    Padding has row -1 and resets its slot before every padded step.
    """
    order = list(range(len(tracks)))
    if shuffle:
        random.Random(seed).shuffle(order)
    pending = iter(order)
    slots = [None] * batch_size
    while True:
        for slot, active in enumerate(slots):
            if active is None:
                track_index = next(pending, None)
                if track_index is not None:
                    slots[slot] = (tracks[track_index], 0)
        if all(active is None for active in slots):
            return
        rows = np.full((batch_size, clip_frames), -1, dtype=np.int64)
        reset = np.ones((batch_size, clip_frames), dtype=bool)
        for slot, active in enumerate(slots):
            if active is None:
                continue
            track, offset = active
            frames = track["frames"]
            stop = min(offset + clip_frames, len(frames))
            for step, position in enumerate(range(offset, stop)):
                frame = frames[position]
                rows[slot, step] = frame["row"]
                reset[slot, step] = (
                    position == 0
                    or frame["frame_index"] != frames[position - 1]["frame_index"] + 1
                )
            slots[slot] = (track, stop) if stop < len(frames) else None
        yield rows, reset


class CachedChunks:
    def __init__(self, cache, batch_size=4, clip_frames=16, shuffle=False, seed=0):
        self.cache = cache
        self.batch_size = batch_size
        self.clip_frames = clip_frames
        self.shuffle = shuffle
        self.seed = seed
        self.frames = [frame for track in cache.index["tracks"] for frame in track["frames"]]

    def __iter__(self):
        for rows, reset in chunk_rows(
            self.cache.index["tracks"], self.batch_size, self.clip_frames,
            self.shuffle, self.seed,
        ):
            valid = rows >= 0
            x = np.zeros((*rows.shape, *self.cache.x.shape[1:]), dtype=np.float32)
            z = np.zeros((*rows.shape, *self.cache.z.shape[1:]), dtype=np.float32)
            x[valid] = self.cache.x[rows[valid]]
            z[valid] = self.cache.z[rows[valid]]
            inout = torch.zeros(rows.shape, dtype=torch.float32)
            targets = torch.zeros((*rows.shape, 64, 64), dtype=torch.float32)
            gaze = np.full((*rows.shape, 2), -1.0, dtype=np.float64)
            for slot, step in zip(*np.nonzero(valid)):
                frame = self.frames[rows[slot, step]]
                inout[slot, step] = frame["inout"]
                gaze[slot, step] = frame["gazex_norm"][0], frame["gazey_norm"][0]
                if frame["inout"] == 1:
                    targets[slot, step] = get_heatmap(*gaze[slot, step], 64, 64)
            yield {
                "x": torch.from_numpy(x), "z": torch.from_numpy(z),
                "reset": torch.from_numpy(reset), "valid": torch.from_numpy(valid),
                "rows": rows, "inout": inout, "heatmap_target": targets, "gaze": gaze,
            }
