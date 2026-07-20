#!/usr/bin/env python3
"""Summarize pre-registered COTB comparisons across independent seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(values: list[float]) -> dict:
    return {
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "values": values,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.comparison]
    primary_names = {payload["primary_subset"] for payload in payloads}
    if len(primary_names) != 1:
        raise ValueError(f"all comparisons must use the same primary subset, got {primary_names}")
    primary = next(iter(primary_names))
    result = {
        "status": "seed_level_descriptive_summary",
        "primary_subset": primary,
        "seed_count": len(payloads),
        "all_seed_go": all(payload["decision"]["verdict"] == "GO" for payload in payloads),
        "control_swap_error": stats([payload["results"][primary]["control_swap_error"]["value"] for payload in payloads]),
        "candidate_swap_error": stats([payload["results"][primary]["candidate_swap_error"]["value"] for payload in payloads]),
        "delta_swap_error": stats([payload["results"][primary]["delta_swap_error"]["value"] for payload in payloads]),
        "delta_diag_margin": stats([payload["results"][primary]["delta_diag_margin"]["value"] for payload in payloads]),
        "l2_delta": stats([payload["l2_delta_candidate_minus_control"] for payload in payloads]),
        "inout_ap_delta": stats([payload["inout_ap_delta_candidate_minus_control"] for payload in payloads]),
        "sources": [str(path.resolve()) for path in args.comparison],
        "note": "Mean/std across training seeds; sequence-cluster bootstrap CIs remain in each source comparison.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_seed_go": result["all_seed_go"], "delta_swap_error": result["delta_swap_error"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
