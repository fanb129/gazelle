"""Export real GF/VAT comparisons and VAT multi-person support/union examples."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cscwd_router.code.stag.final_visuals import export
from cscwd_router.code.stag.engine import seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("gf-summary", "vat-summary", "multi-summary", "gf-data-path", "vat-data-path", "official-checkpoint-dir",
                 "report-root", "figure-root", "provenance"):
        p.add_argument("--" + name, type=Path, required=True)
    for name, default in (("candidate-count", 32), ("examples-per-dataset", 4), ("multi-candidate-count", 16), ("multi-examples", 2), ("seed", 0)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--multi-min-inframe", type=int, choices=(0, 1, 2, 3), default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--allow-closed-gf", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--render", action="store_true")
    return p.parse_args(argv)


def run(args):
    seed_everything(args.seed)
    result = export(args)
    if args.render and not args.check_only:
        from cscwd_router.code.scripts.render_final_visualizations import render
        render(args.report_root / "manifest.json", args.figure_root)
    return result


if __name__ == "__main__":
    run(parse_args())
