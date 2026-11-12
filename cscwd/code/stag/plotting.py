"""Observe saved causal predictions with raw RGB, fixed-scale maps and timelines."""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image

from .cache import file_sha256, write_json


METHODS = ("single_frame", "stag", "ema_alpha_0p25", "ema_alpha_0p5")
LABELS = {"single_frame": "Single frame", "stag": "STAG",
          "ema_alpha_0p25": "EMA alpha=0.25", "ema_alpha_0p5": "EMA alpha=0.5"}
COLORS = {"single_frame": "#1976d2", "stag": "#d32f2f",
          "ema_alpha_0p25": "#388e3c", "ema_alpha_0p5": "#7b1fa2"}


def snapshot_indices(start, stop, anchor, snapshots):
    """Retain the event and its neighbors, then endpoints and uniform context."""
    if snapshots < 1:
        raise ValueError("snapshots must be positive")
    count = min(snapshots, stop - start)
    uniform = np.rint(np.linspace(start, stop - 1, count)).astype(int).tolist()
    selected = []
    for index in [anchor, anchor - 1, anchor + 1, start, stop - 1, *uniform]:
        if start <= index < stop and index not in selected:
            selected.append(index)
        if len(selected) == count:
            break
    return sorted(selected)


def _event_summary(case):
    events = case.get("event")
    if events is None:
        return ""
    names = {"single_frame": "SF", "stag": "STAG", "ema_alpha_0p25": "EMA .25", "ema_alpha_0p5": "EMA .5"}
    parts = [f"Scoring window: {events['single_frame']['window_frames']} frames"]
    for method in METHODS:
        event = events[method]
        outcome = event["status"]
        if outcome == "responded":
            outcome += f" +{event['response_delay_frames']}f"
        parts.append(f"{names[method]}: {outcome}")
    return " | ".join(parts)


def _case_records(case, track, arrays_by_method, diagnostics, heads_by_path):
    records = []
    for global_index in range(case["context_start"], case["context_stop"]):
        frame = track["frames"][global_index - track["start"]]
        gt = [frame["gazex_norm"][0], frame["gazey_norm"][0]] if frame["inout"] == 1 else None
        predictions = {}
        for method in METHODS:
            point = arrays_by_method[method]["argmax_xy"][global_index]
            predictions[method] = {
                "xy": [float(value) for value in point],
                "inout": float(arrays_by_method[method]["inout"][global_index]),
                "error": float(np.linalg.norm(np.asarray(point) - gt)) if gt is not None else None,
            }
        records.append({
            "global_index": global_index, "frame_index": frame["frame_index"], "path": frame["path"],
            "gt_inout": frame["inout"], "gt_xy": gt, "bbox_norm": frame["bbox_norm"],
            "predictions": predictions, "alpha": float(diagnostics["alpha"][global_index]),
            "signals": [float(value) for value in diagnostics["signals"][global_index]],
            "reset": bool(diagnostics["reset"][global_index]),
            "head_count": len(heads_by_path[frame["path"]]),
        })
    return records


def _save_figure(figure, full_path, preview_path, dpi, full_kind, preview_kind):
    figure.savefig(full_path, dpi=dpi, facecolor="white")
    files = [{"path": str(full_path.resolve()), "kind": full_kind,
              "sha256": file_sha256(full_path), "size_bytes": full_path.stat().st_size}]
    if preview_path is not None:
        figure.savefig(preview_path, dpi=70, facecolor="white")
        files.append({"path": str(preview_path.resolve()), "kind": preview_kind,
                      "sha256": file_sha256(preview_path), "size_bytes": preview_path.stat().st_size})
    plt.close(figure)
    return files


def _frame_figure(case, records, snapshots, arrays_by_method, data_path, heads_by_path):
    lookup = {record["global_index"]: record for record in records}
    figure, axes = plt.subplots(len(METHODS), len(snapshots), figsize=(3.0 * len(snapshots), 9.1), squeeze=False)
    figure.subplots_adjust(left=0.065, right=0.90, top=0.84, bottom=0.045, wspace=0.10, hspace=0.38)
    title = (f"{case['category']} | {case['track_id']}\n"
             f"Anchor frame {case['anchor_frame_index']} | saved predictions, offline context")
    event_summary = _event_summary(case)
    figure.suptitle(title + ("\n" + event_summary if event_summary else ""), fontsize=11, y=0.99)
    figure.legend(handles=[Line2D([], [], marker="*", color="none", markerfacecolor="#ffd54f",
                                  markeredgecolor="black", markersize=10, label="GT (in-frame only)"),
                           Line2D([], [], marker="x", color="#d32f2f", linestyle="none", markersize=7,
                                  label="Prediction (row color)"),
                           Line2D([], [], color="#ffb000", linewidth=2, label="Queried head"),
                           Line2D([], [], color="gray", linewidth=1, label="Other annotated heads")],
                  loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=4, fontsize=9, frameon=False)
    for column, global_index in enumerate(snapshots):
        record = lookup[global_index]
        with Image.open(Path(data_path) / record["path"]) as image:
            rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]
        for row, method in enumerate(METHODS):
            axis = axes[row, column]
            axis.imshow(rgb, extent=(0, 1, 1, 0), aspect="auto")
            heatmap_image = axis.imshow(arrays_by_method[method]["heatmap"][global_index],
                                        extent=(0, 1, 1, 0), aspect="auto", cmap="magma",
                                        vmin=0, vmax=1, alpha=0.38, interpolation="nearest")
            for head in heads_by_path[record["path"]]:
                left, top, right, bottom = head["bbox_norm"]
                queried = head["track_id"] == case["track_id"]
                axis.add_patch(Rectangle((left, top), right - left, bottom - top, fill=False,
                                         edgecolor="#ffb000" if queried else "gray",
                                         linewidth=1.8 if queried else 0.8))
            if record["gt_xy"] is not None:
                axis.scatter(*record["gt_xy"], marker="*", s=95, c="#ffd54f", edgecolors="black", linewidths=0.6, zorder=5)
            prediction = record["predictions"][method]
            axis.scatter(*prediction["xy"], marker="x", s=55, c=COLORS[method], linewidths=1.8, zorder=6)
            error = "n/a" if prediction["error"] is None else f"{prediction['error']:.3f}"
            anchor = " [anchor]" if global_index == case["anchor_global_index"] else ""
            axis.set_title(f"Frame {record['frame_index']}{anchor}\np(in)={prediction['inout']:.2f}, L2={error}", fontsize=8)
            if column == 0:
                axis.set_ylabel(LABELS[method], fontsize=9)
            axis.set(xlim=(0, 1), ylim=(1, 0), xticks=[], yticks=[])
            axis.set_box_aspect(height / width)
    color_axis = figure.add_axes([0.92, 0.12, 0.012, 0.65])
    colorbar = figure.colorbar(heatmap_image, cax=color_axis)
    colorbar.set_label("Raw heatmap (fixed 0–1 scale)", fontsize=9)
    return figure


def _timeline_figure(case, records, config):
    frames = np.asarray([record["frame_index"] for record in records])
    gt = np.asarray([record["gt_xy"] if record["gt_xy"] is not None else [np.nan, np.nan] for record in records])
    figure, axes = plt.subplots(6, 1, figsize=(12.5, 11.5), sharex=True)
    figure.subplots_adjust(left=0.085, right=0.98, top=0.89, bottom=0.06, hspace=0.20)
    title = (f"{case['category']} | {case['track_id']}\n"
             "Frame-level cached observations; no prediction recomputation")
    event_summary = _event_summary(case)
    figure.suptitle(title + ("\n" + event_summary if event_summary else ""), fontsize=11, y=0.99)
    for coordinate, axis in enumerate(axes[:2]):
        axis.plot(frames, gt[:, coordinate], color="black", linewidth=1.8, label="GT")
        for method in METHODS:
            points = np.asarray([record["predictions"][method]["xy"] for record in records])
            axis.plot(frames, points[:, coordinate], color=COLORS[method], linewidth=1.1,
                      label=LABELS[method], alpha=0.85)
        axis.set_ylabel("Normalized " + "xy"[coordinate])
        axis.set_ylim(-0.04, 1.04)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5, fontsize=8, frameon=False)
    for method in METHODS:
        errors = [record["predictions"][method]["error"] for record in records]
        axes[2].plot(frames, [np.nan if error is None else error for error in errors],
                     color=COLORS[method], linewidth=1.1, label=LABELS[method])
    axes[2].axhline(config.response_radius, color="black", linestyle=":", linewidth=1,
                    label=f"Response radius={config.response_radius}")
    axes[2].set_ylabel("In-frame L2")
    axes[2].legend(loc="upper right", fontsize=7, frameon=False)
    alpha = np.asarray([record["alpha"] for record in records])
    axes[3].plot(frames, alpha, color=COLORS["stag"], linewidth=1.3, label="STAG alpha")
    reset = np.asarray([record["reset"] for record in records])
    axes[3].scatter(frames[reset], alpha[reset], marker="s", s=20, color="black", label="State reset", zorder=5)
    axes[3].set_ylabel("Alpha")
    axes[3].set_ylim(-0.04, 1.04)
    handles, labels = axes[3].get_legend_handles_labels()
    if case.get("event") is not None:
        handles.append(Patch(facecolor="#fbc02d", alpha=0.14, label="Scoring window"))
        labels.append("Scoring window")
    axes[3].legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    axes[4].plot(frames, [record["predictions"]["single_frame"]["inout"] for record in records],
                 color="#1565c0", label="Raw p(in)")
    axes[4].step(frames, [record["gt_inout"] for record in records], where="mid", color="black",
                 linestyle="--", linewidth=1, label="GT in/out")
    axes[4].axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    axes[4].set_ylabel("In/out")
    axes[4].set_ylim(-0.04, 1.04)
    axes[4].legend(loc="upper right", fontsize=8, frameon=False)
    signals = np.asarray([record["signals"] for record in records])
    for column, (label, color) in enumerate(zip(("d_head", "trans", "d_heat"), ("#00838f", "#ef6c00", "#7b1fa2"))):
        axes[5].plot(frames, signals[:, column], label=label, color=color, linewidth=1.2)
    axes[5].set_ylabel("Gate signals")
    axes[5].legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
    axes[5].set_xlabel("Original frame index")
    for axis in axes:
        if case.get("event") is not None:
            event = case["event"]["single_frame"]
            start = max(frames[0] - 0.5, event["event_frame_index"] - 0.5)
            stop = min(frames[-1] + 0.5, event["event_frame_index"] + event["window_frames"] - 0.5)
            axis.axvspan(start, stop, color="#fbc02d", alpha=0.14, zorder=0)
        axis.axvline(case["anchor_frame_index"], color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        for frame in frames[reset]:
            axis.axvline(frame, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        axis.grid(alpha=0.18)
    return figure


def render_case(case, track, arrays_by_method, diagnostics, data_path, heads_by_path,
                output_dir, config, snapshots=5, dpi=140, preview_dir=None):
    """Render one selected segment, preserving every cached value and GT label."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(preview_dir) if preview_dir is not None else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case["case_id"])
    records = _case_records(case, track, arrays_by_method, diagnostics, heads_by_path)
    snapshots = snapshot_indices(case["context_start"], case["context_stop"], case["anchor_global_index"], snapshots)
    data_file = output_dir / "case_data.json"
    write_json(data_file, {"case": case, "snapshot_global_indices": snapshots,
                           "methods": list(METHODS), "heatmap_display": {"vmin": 0, "vmax": 1, "alpha": 0.38},
                           "frames": records})
    files = [{"path": str(data_file.resolve()), "kind": "data",
              "sha256": file_sha256(data_file), "size_bytes": data_file.stat().st_size}]
    for figure, stem in ((_frame_figure(case, records, snapshots, arrays_by_method, data_path, heads_by_path), "frames"),
                         (_timeline_figure(case, records, config), "timeline")):
        files.extend(_save_figure(figure, output_dir / f"{stem}.png",
                                  preview_dir / f"{safe_name}_{stem}.png" if preview_dir is not None else None,
                                  dpi, f"full_{stem}", f"preview_{stem}"))
    return {"files": files, "frame_global_indices": snapshots, "data_path": str(data_file.resolve())}
