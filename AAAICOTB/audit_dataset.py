#!/usr/bin/env python3
"""Audit whether VAT has enough valid same-frame counterfactual pairs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from AAAICOTB.annotations import load_sequences, sequence_id
from AAAICOTB.geometry import bbox_iou, cluster_targets


def audit(annotation_path: Path, shared_radius: float, min_separation: float, duplicate_iou: float) -> dict:
    sequences = load_sequences(annotation_path)
    counters = Counter()
    separation_values: list[float] = []
    eligible_videos: set[str] = set()
    duplicate_examples: list[dict] = []
    for sequence in sequences:
        for frame in sequence.get("frames", []):
            heads = frame.get("heads", [])
            if not heads:
                continue
            counters["frames"] += 1
            counters["queries"] += len(heads)
            valid = [
                index
                for index, head in enumerate(heads)
                if int(head.get("inout", 0)) == 1
                and float(head["gazex_norm"][0]) >= 0
                and float(head["gazey_norm"][0]) >= 0
            ]
            counters["inframe_queries"] += len(valid)
            if len(valid) < 2:
                continue
            points = [
                (float(heads[index]["gazex_norm"][0]), float(heads[index]["gazey_norm"][0]))
                for index in valid
            ]
            labels, centers = cluster_targets(points, shared_radius)
            counters["multi_inframe_frames"] += 1
            counters["target_clusters"] += len(centers)
            frame_pairs = 0
            for first, second in itertools.combinations(range(len(valid)), 2):
                counters["raw_inframe_pairs"] += 1
                if labels[first] == labels[second]:
                    counters["shared_target_pairs"] += 1
                    continue
                separation = math.dist(centers[labels[first]], centers[labels[second]])
                if separation < min_separation:
                    counters["too_close_pairs"] += 1
                    continue
                first_global, second_global = valid[first], valid[second]
                iou = bbox_iou(heads[first_global]["bbox_norm"], heads[second_global]["bbox_norm"])
                if iou >= duplicate_iou:
                    counters["duplicate_bbox_pairs"] += 1
                    if len(duplicate_examples) < 20:
                        duplicate_examples.append(
                            {"path": frame["path"], "query_i": first_global, "query_j": second_global, "iou": iou}
                        )
                    continue
                frame_pairs += 1
                separation_values.append(separation)
                counters["eligible_pairs"] += 1
                if separation >= 0.30:
                    counters["far_pairs"] += 1
                if len(heads) >= 4:
                    counters["crowd_ge4_pairs"] += 1
                if len(heads) >= 5:
                    counters["crowd_ge5_pairs"] += 1
                if separation >= 0.30 and len(heads) >= 4:
                    counters["far_crowd_ge4_pairs"] += 1
            if frame_pairs:
                counters["eligible_frames"] += 1
                eligible_videos.add(sequence_id(sequence, frame))

    ordered = dict(sorted(counters.items()))
    support_ok = ordered.get("far_pairs", 0) >= 1000 and len(eligible_videos) >= 30
    return {
        "status": "measured_annotation_audit",
        "annotation": str(annotation_path.resolve()),
        "definitions": {
            "shared_target_radius": shared_radius,
            "min_pair_separation": min_separation,
            "duplicate_bbox_iou": duplicate_iou,
            "far_pair_separation": 0.30,
        },
        "counts": ordered,
        "eligible_sequence_count": len(eligible_videos),
        "eligible_sequences": sorted(eligible_videos),
        "separation_mean": sum(separation_values) / len(separation_values) if separation_values else None,
        "support_gate": {
            "pass": support_ok,
            "requirements": "at least 1000 far pairs and 30 eligible VAT sequences",
        },
        "duplicate_bbox_examples": duplicate_examples,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-target-radius", type=float, default=0.06)
    parser.add_argument("--min-pair-separation", type=float, default=0.10)
    parser.add_argument("--duplicate-bbox-iou", type=float, default=0.80)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    report = audit(args.annotation, args.shared_target_radius, args.min_pair_separation, args.duplicate_bbox_iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "support_gate": report["support_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
