"""Report whether a GazeFollow annotation split contains multi-head images."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import posixpath
from pathlib import Path
from typing import Iterable, Optional, Sequence

from gazelle.data_splits import grouped_holdout_split


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit per-image query counts in preprocessed GazeFollow JSON."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--val_fraction", type=float, default=0.0)
    parser.add_argument("--split_seed", type=int, default=3106)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _normalized_path(record: dict, index: int) -> str:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"record {index} has no valid image path")
    return posixpath.normpath(raw_path.strip().replace("\\", "/"))


def _nearest_rank(values: Sequence[int], fraction: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return int(ordered[index])


def query_statistics(records: Sequence[dict], indices: Sequence[int]) -> dict:
    all_counts = []
    inframe_counts = []
    multi_head_examples = []
    for index in indices:
        heads = records[index].get("heads") or []
        all_count = len(heads)
        inframe_count = sum(int(head.get("inout", 1) == 1) for head in heads)
        all_counts.append(all_count)
        inframe_counts.append(inframe_count)
        if inframe_count > 1 and len(multi_head_examples) < 20:
            multi_head_examples.append(
                {
                    "record_index": int(index),
                    "path": _normalized_path(records[index], index),
                    "inframe_head_count": inframe_count,
                }
            )

    histogram = Counter(inframe_counts)
    return {
        "record_count": len(indices),
        "total_head_count": sum(all_counts),
        "total_inframe_head_count": sum(inframe_counts),
        "zero_inframe_record_count": histogram.get(0, 0),
        "single_head_record_count": histogram.get(1, 0),
        "multi_head_record_count": sum(
            count for head_count, count in histogram.items() if head_count > 1
        ),
        "multi_head_person_count": sum(
            head_count * count
            for head_count, count in histogram.items()
            if head_count > 1
        ),
        "max_inframe_heads_per_image": max(inframe_counts, default=0),
        "p50_inframe_heads_per_image": _nearest_rank(inframe_counts, 0.50),
        "p90_inframe_heads_per_image": _nearest_rank(inframe_counts, 0.90),
        "p95_inframe_heads_per_image": _nearest_rank(inframe_counts, 0.95),
        "p99_inframe_heads_per_image": _nearest_rank(inframe_counts, 0.99),
        "inframe_head_count_histogram": {
            str(head_count): count
            for head_count, count in sorted(histogram.items())
        },
        "multi_head_examples": multi_head_examples,
    }


def main(argv: Optional[Iterable[str]] = None) -> dict:
    args = parse_args(argv)
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val_fraction must be in [0, 1)")

    annotation_path = (
        Path(args.data_path) / f"{args.split}_preprocessed.json"
    )
    records = json.loads(annotation_path.read_text(encoding="utf-8"))
    group_keys = [
        _normalized_path(record, index)
        for index, record in enumerate(records)
    ]
    duplicates = len(group_keys) - len(set(group_keys))
    result = {
        "annotation_file": str(annotation_path.resolve()),
        "split": args.split,
        "unique_image_path_count": len(set(group_keys)),
        "duplicate_image_path_record_count": duplicates,
        "all": query_statistics(records, tuple(range(len(records)))),
    }

    if args.val_fraction:
        split = grouped_holdout_split(
            group_keys,
            validation_fraction=args.val_fraction,
            seed=args.split_seed,
        )
        result["holdout"] = {
            **split.metadata(),
            "train": query_statistics(records, split.train_indices),
            "validation": query_statistics(records, split.validation_indices),
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved query-group audit to {output_path}")
    return result


if __name__ == "__main__":
    main()
