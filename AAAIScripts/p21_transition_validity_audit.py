#!/usr/bin/env python3
"""Generate and summarize a manual validity audit for VAT in/out transitions.

Generation is deterministic and uses only existing P2 transition records, VAT
annotations, and RGB frames.  It does not load a model or checkpoint.  Reviewers
fill the four ``review_*`` columns in ``audit.csv`` and then run summarize mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIScripts.failure_taxonomy import flatten_annotations
from AAAIScripts.p2_transition_reliability_audit import bbox_iou, clustered_delta


AUDIT_FIELDS = [
    "sample_id",
    "state",
    "sequence_id",
    "previous_path",
    "current_path",
    "previous_person_index",
    "current_person_index",
    "bbox_iou",
    "crowd_bin",
    "head_size_bin",
    "baseline_v0_448_score",
    "learned_sasa_ggsf_512_score",
    "p1a_static_prior_512_score",
    "p1a_prior_residual_512_score",
    "sheet_path",
    "review_valid_transition",
    "review_temporal_context_helpful",
    "review_issue_type",
    "review_notes",
]

VALIDITY_CHOICES = {"yes", "no", "uncertain"}
HELPFUL_CHOICES = {"yes", "no", "uncertain"}
ISSUE_CHOICES = {
    "clear_transition",
    "gradual_ambiguous",
    "annotation_inconsistency",
    "track_mismatch",
    "occlusion_scene_cut",
    "other",
}
INVALID_ISSUES = {"annotation_inconsistency", "track_mismatch"}
MODEL_LABELS = (
    "baseline_v0_448",
    "learned_sasa_ggsf_512",
    "p1a_static_prior_512",
    "p1a_prior_residual_512",
)


def transition_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        row["previous_path"],
        row["current_path"],
        int(row["previous_person_index"]),
        int(row["current_person_index"]),
    )


def sample_id(key: tuple[str, str, int, int]) -> str:
    payload = "|".join(map(str, key)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def read_transition_groups(path: Path) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = transition_key(row)
            item = groups.setdefault(
                key,
                {
                    "key": key,
                    "state": row["state"],
                    "sequence_id": row["sequence_id"],
                    "bbox_iou": float(row["bbox_iou"]),
                    "crowd_bin": row["crowd_bin"],
                    "head_size_bin": row["head_size_bin"],
                    "scores": {},
                },
            )
            if item["state"] != row["state"] or item["sequence_id"] != row["sequence_id"]:
                raise ValueError(f"inconsistent duplicate transition: {key}")
            model = row["model"]
            if model in item["scores"]:
                raise ValueError(f"duplicate model row for transition: {model}, {key}")
            item["scores"][model] = float(row["current_score"])
    for key, item in groups.items():
        missing = sorted(set(MODEL_LABELS) - set(item["scores"]))
        if missing:
            raise ValueError(f"transition {key} is missing models: {missing}")
    return groups


def stratified_sample(
    items: list[dict[str, Any]],
    per_direction: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for state in ("in_to_out", "out_to_in"):
        candidates = [item for item in items if item["state"] == state]
        if len(candidates) < per_direction:
            raise ValueError(f"only {len(candidates)} candidates for {state}, need {per_direction}")
        by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in candidates:
            by_sequence[item["sequence_id"]].append(item)
        sequences = sorted(by_sequence)
        rng.shuffle(sequences)
        for sequence in sequences:
            rng.shuffle(by_sequence[sequence])
        state_selected: list[dict[str, Any]] = []
        depth = 0
        while len(state_selected) < per_direction:
            added = False
            for sequence in sequences:
                bucket = by_sequence[sequence]
                if depth < len(bucket):
                    state_selected.append(bucket[depth])
                    added = True
                    if len(state_selected) == per_direction:
                        break
            if not added:
                raise RuntimeError(f"unable to finish sampling for {state}")
            depth += 1
        selected.extend(state_selected)
    rng.shuffle(selected)
    return selected


def frame_number(path: str) -> int | None:
    try:
        return int(Path(path).stem)
    except ValueError:
        return None


def build_annotation_index(payload: Any):
    frames = flatten_annotations("vat", payload)
    frame_by_path = {frame["path"]: frame for frame in frames}
    if len(frame_by_path) != len(frames):
        raise ValueError("duplicate frame paths in VAT annotation")
    sequences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        sequences[str(Path(frame["path"]).parent)].append(frame)
    for sequence in sequences.values():
        sequence.sort(
            key=lambda frame: (
                frame_number(frame["path"]) is None,
                frame_number(frame["path"]) or 0,
            )
        )
    return frame_by_path, sequences


def best_iou_match(reference: dict[str, Any], candidates: list[dict[str, Any]], threshold: float):
    if reference.get("bbox_norm") is None:
        return None
    scored = [
        (bbox_iou(reference["bbox_norm"], candidate["bbox_norm"]), index)
        for index, candidate in enumerate(candidates)
        if candidate.get("bbox_norm") is not None
    ]
    if not scored:
        return None
    score, index = max(scored)
    return index if score >= threshold else None


def context_frames(
    item: dict[str, Any],
    sequences: dict[str, list[dict[str, Any]]],
    iou_threshold: float,
) -> list[tuple[str, dict[str, Any] | None, int | None]]:
    previous_path, current_path, previous_index, current_index = item["key"]
    frames = sequences[item["sequence_id"]]
    positions = {frame["path"]: index for index, frame in enumerate(frames)}
    if previous_path not in positions or current_path not in positions:
        raise ValueError(f"transition paths missing from sequence: {item['key']}")
    previous_position, current_position = positions[previous_path], positions[current_path]
    if current_position != previous_position + 1:
        raise ValueError(f"transition is not adjacent in sorted annotation: {item['key']}")
    previous_frame, current_frame = frames[previous_position], frames[current_position]
    if previous_index >= len(previous_frame["heads"]) or current_index >= len(current_frame["heads"]):
        raise ValueError(f"person index outside annotation: {item['key']}")

    earlier_frame = frames[previous_position - 1] if previous_position > 0 else None
    earlier_index = None
    if earlier_frame is not None:
        earlier_index = best_iou_match(
            previous_frame["heads"][previous_index], earlier_frame.get("heads", []), iou_threshold
        )
    later_frame = frames[current_position + 1] if current_position + 1 < len(frames) else None
    later_index = None
    if later_frame is not None:
        later_index = best_iou_match(
            current_frame["heads"][current_index], later_frame.get("heads", []), iou_threshold
        )
    return [
        ("t-2", earlier_frame, earlier_index),
        ("t-1", previous_frame, previous_index),
        ("t", current_frame, current_index),
        ("t+1", later_frame, later_index),
    ]


def normalized_target(head: dict[str, Any]) -> tuple[float, float] | None:
    if int(head.get("inout", 0)) != 1:
        return None
    xs = [float(value) for value in (head.get("gazex_norm") or []) if float(value) >= 0]
    ys = [float(value) for value in (head.get("gazey_norm") or []) if float(value) >= 0]
    if not xs or not ys:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def draw_panel(
    data_root: Path,
    frame: dict[str, Any] | None,
    person_index: int | None,
    label: str,
    panel_width: int = 360,
    image_height: int = 250,
) -> Image.Image:
    header_height = 54
    panel = Image.new("RGB", (panel_width, header_height + image_height), "white")
    draw = ImageDraw.Draw(panel)
    if frame is None:
        draw.text((8, 8), f"{label}: unavailable", fill="black")
        return panel
    image_path = data_root / frame["path"]
    if not image_path.is_file():
        raise FileNotFoundError(f"VAT frame not found: {image_path}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    source_width, source_height = image.size
    scale = min(panel_width / source_width, image_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    x_offset = (panel_width - resized_width) // 2
    y_offset = header_height + (image_height - resized_height) // 2
    panel.paste(resized, (x_offset, y_offset))
    short_path = "/".join(Path(frame["path"]).parts[-2:])
    if person_index is None or person_index >= len(frame.get("heads", [])):
        draw.text((8, 6), f"{label}: no IoU match", fill="black")
        draw.text((8, 25), short_path, fill="black")
        return panel
    head = frame["heads"][person_index]
    inout = int(head.get("inout", 0))
    draw.text((8, 5), f"{label} | person={person_index} | inout={inout}", fill="black")
    draw.text((8, 25), short_path, fill="black")
    x1, y1, x2, y2 = map(float, head["bbox_norm"])
    box = (
        x_offset + x1 * resized_width,
        y_offset + y1 * resized_height,
        x_offset + x2 * resized_width,
        y_offset + y2 * resized_height,
    )
    draw.rectangle(box, outline=(255, 45, 45), width=4)
    target = normalized_target(head)
    if target is not None:
        tx = x_offset + target[0] * resized_width
        ty = y_offset + target[1] * resized_height
        hx = (box[0] + box[2]) / 2
        hy = (box[1] + box[3]) / 2
        draw.line((hx, hy, tx, ty), fill=(0, 235, 255), width=3)
        radius = 7
        draw.ellipse((tx - radius, ty - radius, tx + radius, ty + radius), fill=(0, 235, 255))
    return panel


def create_contact_sheet(
    item: dict[str, Any],
    data_root: Path,
    sequences: dict[str, list[dict[str, Any]]],
    output_path: Path,
    iou_threshold: float,
) -> None:
    panels = [
        draw_panel(data_root, frame, person_index, label)
        for label, frame, person_index in context_frames(item, sequences, iou_threshold)
    ]
    top_height, bottom_height = 46, 104
    width = sum(panel.width for panel in panels)
    height = top_height + panels[0].height + bottom_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    identifier = sample_id(item["key"])
    draw.text(
        (10, 8),
        f"sample={identifier} | {item['state']} | bbox_iou={item['bbox_iou']:.3f} "
        f"| crowd={item['crowd_bin']} | head={item['head_size_bin']}",
        fill="black",
    )
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, top_height))
        x += panel.width
    score_lines = [
        "Current-frame in-score:",
        f"baseline_v0_448={item['scores']['baseline_v0_448']:.4f}    "
        f"learned_sasa_ggsf_512={item['scores']['learned_sasa_ggsf_512']:.4f}",
        f"p1a_static_prior_512={item['scores']['p1a_static_prior_512']:.4f}    "
        f"p1a_prior_residual_512={item['scores']['p1a_prior_residual_512']:.4f}",
        "Red=bbox, cyan=GT gaze target when in-frame. Review the label boundary, not model correctness.",
    ]
    y = top_height + panels[0].height + 7
    for line in score_lines:
        draw.text((10, y), line, fill="black")
        y += 21
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=88, optimize=True)


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def generate(args: argparse.Namespace) -> None:
    audit_path = args.output_dir / "audit.csv"
    if audit_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing manual-audit table: {audit_path}"
        )
    groups = read_transition_groups(args.transition_csv)
    candidates = [item for item in groups.values() if item["state"] in {"in_to_out", "out_to_in"}]
    selected = stratified_sample(candidates, args.per_direction, args.seed)
    payload = json.loads(args.annotation_json.read_text(encoding="utf-8"))
    _, sequences = build_annotation_index(payload)
    sheet_dir = args.output_dir / "sheets"
    audit_rows = []
    for index, item in enumerate(selected, start=1):
        identifier = sample_id(item["key"])
        sheet_name = f"{index:03d}_{item['state']}_{identifier}.jpg"
        create_contact_sheet(
            item, args.data_root, sequences, sheet_dir / sheet_name, args.iou_threshold
        )
        previous_path, current_path, previous_index, current_index = item["key"]
        audit_rows.append(
            {
                "sample_id": identifier,
                "state": item["state"],
                "sequence_id": item["sequence_id"],
                "previous_path": previous_path,
                "current_path": current_path,
                "previous_person_index": previous_index,
                "current_person_index": current_index,
                "bbox_iou": item["bbox_iou"],
                "crowd_bin": item["crowd_bin"],
                "head_size_bin": item["head_size_bin"],
                **{f"{model}_score": item["scores"][model] for model in MODEL_LABELS},
                "sheet_path": f"sheets/{sheet_name}",
                "review_valid_transition": "",
                "review_temporal_context_helpful": "",
                "review_issue_type": "",
                "review_notes": "",
            }
        )
    write_audit_csv(audit_path, audit_rows)
    manifest = {
        "status": "awaiting_manual_review",
        "seed": args.seed,
        "per_direction": args.per_direction,
        "sample_count": len(audit_rows),
        "state_counts": {
            state: sum(row["state"] == state for row in audit_rows)
            for state in ("in_to_out", "out_to_in")
        },
        "sequence_count": len({row["sequence_id"] for row in audit_rows}),
        "review_columns": {
            "review_valid_transition": sorted(VALIDITY_CHOICES),
            "review_temporal_context_helpful": sorted(HELPFUL_CHOICES),
            "review_issue_type": sorted(ISSUE_CHOICES),
            "review_notes": "free text",
        },
        "frozen_gates": {
            "max_invalid_rate": 0.10,
            "min_temporal_context_helpful_rate": 0.70,
            "effect_requirement": "both direction-matched absolute-error CI lows > 0 for >=3 models including >=2 512 models",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(audit_rows)} contact sheets and {audit_path}")


def normalized_choice(value: str) -> str:
    return value.strip().lower()


def read_completed_audit(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("audit CSV is empty")
    seen = set()
    for line_number, row in enumerate(rows, start=2):
        identifier = row.get("sample_id", "")
        if not identifier or identifier in seen:
            raise ValueError(f"missing or duplicate sample_id at CSV line {line_number}")
        seen.add(identifier)
        valid = normalized_choice(row.get("review_valid_transition", ""))
        helpful = normalized_choice(row.get("review_temporal_context_helpful", ""))
        issue = normalized_choice(row.get("review_issue_type", ""))
        if valid not in VALIDITY_CHOICES:
            raise ValueError(f"invalid review_valid_transition at line {line_number}: {valid!r}")
        if helpful not in HELPFUL_CHOICES:
            raise ValueError(f"invalid review_temporal_context_helpful at line {line_number}: {helpful!r}")
        if issue not in ISSUE_CHOICES:
            raise ValueError(f"invalid review_issue_type at line {line_number}: {issue!r}")
        row["review_valid_transition"] = valid
        row["review_temporal_context_helpful"] = helpful
        row["review_issue_type"] = issue
    return rows


def numeric_transition_rows(path: Path) -> list[dict[str, Any]]:
    numeric_fields = {
        "bbox_iou",
        "brier",
        "absolute_calibration_error",
        "current_l2",
        "current_score",
        "previous_score",
        "score_step",
    }
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for field in numeric_fields:
                if row.get(field) not in (None, ""):
                    row[field] = float(row[field])
                else:
                    row[field] = None
            row["previous_person_index"] = int(row["previous_person_index"])
            row["current_person_index"] = int(row["current_person_index"])
            rows.append(row)
    return rows


def summarize(args: argparse.Namespace) -> None:
    audit_rows = read_completed_audit(args.audit_csv)
    expected_count = 2 * args.per_direction
    if len(audit_rows) != expected_count:
        raise ValueError(f"expected {expected_count} audit rows, found {len(audit_rows)}")
    state_counts = {
        state: sum(row["state"] == state for row in audit_rows)
        for state in ("in_to_out", "out_to_in")
    }
    if any(count != args.per_direction for count in state_counts.values()):
        raise ValueError(f"audit state counts changed: {state_counts}")

    invalid_rows = [
        row
        for row in audit_rows
        if row["review_valid_transition"] != "yes" or row["review_issue_type"] in INVALID_ISSUES
    ]
    helpful_rows = [
        row for row in audit_rows if row["review_temporal_context_helpful"] == "yes"
    ]
    invalid_rate = len(invalid_rows) / len(audit_rows)
    helpful_rate = len(helpful_rows) / len(audit_rows)
    gate_validity = invalid_rate <= 0.10
    gate_helpful = helpful_rate >= 0.70

    valid_keys = {
        (
            row["previous_path"],
            row["current_path"],
            int(row["previous_person_index"]),
            int(row["current_person_index"]),
        )
        for row in audit_rows
        if row["review_valid_transition"] == "yes" and row["review_issue_type"] not in INVALID_ISSUES
    }
    transition_rows = numeric_transition_rows(args.transition_csv)
    comparisons = []
    model_passes = {}
    for model_index, model in enumerate(sorted({row["model"] for row in transition_rows})):
        model_rows = [row for row in transition_rows if row["model"] == model]
        results = []
        for comparison_index, (state, control) in enumerate(
            (("in_to_out", "stable_out"), ("out_to_in", "stable_in"))
        ):
            result = clustered_delta(
                model_rows,
                lambda row, state=state: row["state"] == state
                and transition_key(row) in valid_keys,
                lambda row, control=control: row["state"] == control,
                "absolute_calibration_error",
                args.bootstrap_iterations,
                args.seed + model_index * 10 + comparison_index,
            )
            result_row = {
                "model": model,
                "comparison": f"reviewed_valid_{state}_minus_{control}_absolute_error",
                **result,
            }
            comparisons.append(result_row)
            results.append(result_row)
        model_passes[model] = all(
            result["ci95_low"] is not None and result["ci95_low"] > 0 for result in results
        )
    passing_models = [model for model, passed in model_passes.items() if passed]
    passing_512_models = [model for model in passing_models if model.endswith("_512")]
    gate_effect = len(passing_models) >= 3 and len(passing_512_models) >= 2
    outcome = "GO" if gate_validity and gate_helpful and gate_effect else "NO_GO"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "filtered_comparisons.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({field for row in comparisons for field in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)
    report = {
        "status": "manual_review_summarized",
        "outcome": outcome,
        "reviewed_count": len(audit_rows),
        "state_counts": state_counts,
        "valid_count": len(audit_rows) - len(invalid_rows),
        "invalid_count": len(invalid_rows),
        "invalid_rate": invalid_rate,
        "temporal_context_helpful_count": len(helpful_rows),
        "temporal_context_helpful_rate": helpful_rate,
        "gates": {
            "invalid_rate_le_0.10": gate_validity,
            "temporal_context_helpful_rate_ge_0.70": gate_helpful,
            "direction_matched_effect_in_3_models_including_2_512": gate_effect,
        },
        "passing_models": passing_models,
        "comparisons": comparisons,
        "decision_rule": "GO only when all three frozen gates pass.",
    }
    report_path = args.output_dir / "review_summary.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"outcome": outcome, "gates": report["gates"]}, indent=2))
    print(f"Wrote {report_path} and {comparison_path}")


def run_self_test() -> None:
    items = []
    for state in ("in_to_out", "out_to_in"):
        for index in range(6):
            items.append(
                {
                    "state": state,
                    "sequence_id": f"seq{index % 3}",
                    "key": (f"{state}_p{index}", f"{state}_c{index}", 0, 0),
                }
            )
    selected = stratified_sample(items, 4, 3106)
    assert len(selected) == 8
    assert sum(item["state"] == "in_to_out" for item in selected) == 4
    assert len({sample_id(item["key"]) for item in selected}) == 8
    assert math.isclose(bbox_iou([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        image_path = root / "images" / "show" / "clip" / "0001.jpg"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (320, 180), "gray").save(image_path)
        frame = {
            "path": "images/show/clip/0001.jpg",
            "heads": [
                {
                    "bbox_norm": [0.1, 0.1, 0.3, 0.4],
                    "inout": 1,
                    "gazex_norm": [0.8],
                    "gazey_norm": [0.6],
                }
            ],
        }
        panel = draw_panel(root, frame, 0, "t")
        assert panel.size == (360, 304)
    print("Self-test passed: deterministic sampling, IDs, IoU, and contact-sheet drawing.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--data-root", type=Path, required=True)
    generate_parser.add_argument("--annotation-json", type=Path, required=True)
    generate_parser.add_argument("--transition-csv", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--per-direction", type=int, default=60)
    generate_parser.add_argument("--iou-threshold", type=float, default=0.3)
    generate_parser.add_argument("--seed", type=int, default=3106)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--transition-csv", type=Path, required=True)
    summarize_parser.add_argument("--audit-csv", type=Path, required=True)
    summarize_parser.add_argument("--output-dir", type=Path, required=True)
    summarize_parser.add_argument("--per-direction", type=int, default=60)
    summarize_parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    summarize_parser.add_argument("--seed", type=int, default=3106)

    subparsers.add_parser("self-test")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "generate":
        generate(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        run_self_test()


if __name__ == "__main__":
    main()
