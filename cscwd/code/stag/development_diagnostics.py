"""Train-only R1 diagnostic sampling and RGB/head contact sheets.

Categories describe observable proxies, not human-verified attention changes.
Past and future display context is used only for offline diagnosis; predictions
are read from the frozen single-frame cache without modification.
"""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


CATEGORY_ORDER = ("gaze_shift_proxy", "state_transition", "prediction_jump", "stable_reference")
LABEL_CHOICES = ("same_target_motion", "camera_change", "attention_target_change",
                 "prediction_error", "unresolvable")


def select_development_cases(tracks, argmax_xy, inout, config, cases_per_category=3,
                             seed=20261117, context_before=10, context_after=20):
    """Sample supplied development-train tracks with one case per clip/category.

    Track slices retain their original indices into the full prediction arrays.
    The caller supplies only the training partition, never the held-out subset.
    """
    if any(not track["track_id"].startswith("train/") for track in tracks):
        raise ValueError("Development diagnostics require train tracks only")
    candidates = {category: [] for category in CATEGORY_ORDER}

    def add_case(category, track, position, reason):
        frames = track["frames"]
        left, right = position, position + 1
        while left > 0 and frames[left]["frame_index"] - frames[left - 1]["frame_index"] == 1:
            left -= 1
        while right < len(frames) and frames[right]["frame_index"] - frames[right - 1]["frame_index"] == 1:
            right += 1
        first = max(left, position - context_before)
        stop = min(right, position + context_after + 1)
        frame_index = frames[position]["frame_index"]
        identity = f"{category}\0{track['track_id']}\0{frame_index}"
        candidates[category].append({
            "case_id": f"{category}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
            "category": category, "track_id": track["track_id"],
            "sequence_path": track["sequence_path"],
            "anchor_global_index": int(track["start"] + position),
            "anchor_frame_index": int(frame_index),
            "context_start": int(track["start"] + first),
            "context_stop": int(track["start"] + stop),
            "frame_paths": [frame["path"] for frame in frames[first:stop]],
            "selection_reason": reason,
            "review_status": "unreviewed", "labels": [], "notes": "",
        })

    for track in tracks:
        frames = track["frames"]
        indices = np.asarray([frame["frame_index"] for frame in frames])
        inside = np.asarray([frame["inout"] == 1 for frame in frames])
        truth = np.asarray([[frame["gazex_norm"][0], frame["gazey_norm"][0]] for frame in frames])
        predicted = np.asarray(argmax_xy[track["start"]:track["stop"]], dtype=np.float64)
        adjacent = np.diff(indices) == 1
        gt_steps = np.linalg.norm(np.diff(truth, axis=0), axis=1)
        prediction_steps = np.linalg.norm(np.diff(predicted, axis=0), axis=1)
        both_inside = adjacent & inside[:-1] & inside[1:]
        masks = {
            "gaze_shift_proxy": both_inside & (gt_steps > config.shift_threshold),
            "state_transition": adjacent & (inside[:-1] != inside[1:]),
            "prediction_jump": both_inside & (gt_steps <= config.stable_threshold)
                               & (prediction_steps > config.shift_threshold),
        }
        reasons = {
            "gaze_shift_proxy": "Adjacent GT-inframe points exceed the shift threshold; not a verified target change",
            "state_transition": "Adjacent ground-truth in/out states differ",
            "prediction_jump": "Prediction jumps while adjacent GT-inframe points remain stable",
        }
        for category, mask in masks.items():
            for position in np.flatnonzero(mask) + 1:
                add_case(category, track, int(position), reasons[category])

        previous_hit = False
        for end in range(config.hold_frames - 1, len(frames)):
            start = end - config.hold_frames + 1
            hit = (inside[start:end + 1].all() and adjacent[start:end].all()
                   and (gt_steps[start:end] <= config.stable_threshold).all())
            if hit and not previous_hit:
                add_case("stable_reference", track, end,
                         "First eligible GT-stable window of an episode; prediction quality is unrestricted")
            previous_hit = bool(hit)

    selected, counts = [], {}
    for category in CATEGORY_ORDER:
        ranked = sorted(candidates[category], key=lambda case: (
            hashlib.sha256(f"{seed}\0{case['case_id']}".encode()).hexdigest(), case["case_id"]))
        clips = {}
        for case in ranked:
            clips.setdefault(case["sequence_path"], case)
        chosen = list(clips.values())[:cases_per_category]
        selected.extend(chosen)
        counts[category] = {"candidates": len(ranked), "candidate_clips": len(clips),
                            "selected": len(chosen)}
    return {
        "protocol": {
            "version": "r1_train_diagnostics_v1", "partition": "development_train_only",
            "category_order": list(CATEGORY_ORDER), "seed": seed,
            "ranking": "seeded_SHA256_of_case_identity; not_error_size_or_model_gain",
            "sampling_unit": "at_most_one_case_per_sequence_path_per_category",
            "cases_per_category": cases_per_category,
            "shift_threshold": config.shift_threshold, "stable_threshold": config.stable_threshold,
            "hold_frames": config.hold_frames,
            "stable_episode": "first_GT_stable_window_of_consecutive_eligible_window_end_frames",
            "context_before": context_before, "context_after": context_after,
            "context": "offline_display_only; clipped_to_same_track_contiguous_original_frames",
            "label_choices": list(LABEL_CHOICES),
            "manual_review": "unreviewed_until_RGB_review; proxy_categories_are_not_manual_labels",
            "empty_categories": "zero_counts; no_replacement_or_threshold_relaxation",
        },
        "counts": counts, "selected": selected,
    }


def _snapshot_indices(case, snapshots):
    first, stop = case["context_start"], case["context_stop"]
    anchor = case["anchor_global_index"]
    ordered = [anchor, anchor - 1, anchor + 1, first, stop - 1]
    ordered.extend(int(value) for value in np.linspace(first, stop - 1, snapshots))
    chosen = []
    for index in ordered:
        if first <= index < stop and index not in chosen:
            chosen.append(index)
        if len(chosen) == snapshots:
            break
    return sorted(chosen)


def render_development_case(case, track, argmax_xy, inout, data_path, output_path, snapshots=5):
    """Render frozen RGB, GT heads/points, predictions and enlarged head crops."""
    indices = _snapshot_indices(case, snapshots)
    column_width, header, frame_height, head_height, footer = 340, 68, 250, 225, 48
    canvas = Image.new("RGB", (column_width * len(indices), header + frame_height + head_height + footer), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 8), f"{case['category']} | anchor={case['anchor_frame_index']} | {case['track_id']}",
              fill="black", font=font)
    draw.text((12, 27), "GT head: blue | GT gaze: green ring (inside only) | raw prediction: red cross",
              fill="black", font=font)
    draw.text((12, 46), "Offline diagnostic context; human labels remain unreviewed.", fill="black", font=font)
    frame_indices = []
    for column, index in enumerate(indices):
        frame = track["frames"][index - track["start"]]
        frame_indices.append(int(frame["frame_index"]))
        with Image.open(Path(data_path) / frame["path"]) as image:
            original = image.convert("RGB")
        shown = ImageOps.contain(original, (column_width - 20, 210), Image.Resampling.LANCZOS)
        panel = ImageDraw.Draw(shown)
        box = frame["bbox_norm"]
        panel.rectangle((box[0] * shown.width, box[1] * shown.height,
                         box[2] * shown.width, box[3] * shown.height), outline="blue", width=2)
        if frame["inout"] == 1:
            x, y = frame["gazex_norm"][0] * shown.width, frame["gazey_norm"][0] * shown.height
            panel.ellipse((x - 6, y - 6, x + 6, y + 6), outline="green", width=2)
        x, y = np.asarray(argmax_xy[index]) * (shown.width, shown.height)
        panel.line((x - 5, y - 5, x + 5, y + 5), fill="red", width=2)
        panel.line((x - 5, y + 5, x + 5, y - 5), fill="red", width=2)
        left = column * column_width
        draw.text((left + 10, header + 6),
                  f"frame={frame['frame_index']} GT={'in' if frame['inout'] else 'out'} s={float(inout[index]):.3f}",
                  fill="black", font=font)
        canvas.paste(shown, (left + (column_width - shown.width) // 2, header + 28))
        crop = original.crop((int(box[0] * original.width), int(box[1] * original.height),
                              int(box[2] * original.width), int(box[3] * original.height)))
        crop = ImageOps.contain(crop, (200, 190), Image.Resampling.LANCZOS)
        head_top = header + frame_height
        draw.text((left + 10, head_top + 5), "Current-frame GT head crop", fill="black", font=font)
        canvas.paste(crop, (left + (column_width - crop.width) // 2, head_top + 24))
    draw.text((12, canvas.height - footer + 6),
              "Prediction point is shown for all states; only GT-inframe labels supply a GT point.",
              fill="black", font=font)
    draw.text((12, canvas.height - footer + 24),
              "Review labels: same_target_motion / camera_change / attention_target_change / prediction_error / unresolvable",
              fill="black", font=font)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return {"path": str(output_path.resolve()), "width": canvas.width, "height": canvas.height,
            "snapshot_global_indices": indices, "snapshot_frame_indices": frame_indices}
