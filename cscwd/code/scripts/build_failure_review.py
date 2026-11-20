"""Export train AND test manual review cases from existing GazeSpot predictions."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from cscwd.code.stag.review_package import build_review_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--provenance", default=str(REPO_ROOT / "cscwd/results/RESULT_PROVENANCE.md"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-frames", type=int, default=16)
    parser.add_argument("--max-history-frames", type=int, default=64)
    parser.add_argument("--large-l2-cases", type=int, default=16)
    parser.add_argument("--fn-cases", type=int, default=8)
    parser.add_argument("--fp-cases", type=int, default=8)
    parser.add_argument("--medium-cases", type=int, default=8)
    parser.add_argument("--control-cases", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--large-l2-quantile", type=float, default=0.9)
    parser.add_argument("--max-per-track", type=int, default=2)
    parser.add_argument("--max-per-video", type=int, default=6)
    parser.add_argument("--image-max-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values["counts"] = {
        "large_l2": values.pop("large_l2_cases"),
        "inout_false_negative": values.pop("fn_cases"),
        "inout_false_positive": values.pop("fp_cases"),
        "medium_l2": values.pop("medium_cases"),
        "correct_control": values.pop("control_cases"),
    }
    build_review_package(**values)


if __name__ == "__main__":
    main()
