"""Run one conditional GF or GF-to-VAT training phase with the original recipe."""

import argparse
import functools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cscwd_router.code.stag.conditional import CONDITIONS, get_conditional_model
from cscwd_router.code.stag.checkpoint import sha256


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--phase", choices=("gf", "vat"), required=True)
    selector.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    selected, remaining = selector.parse_known_args(argv)
    if selected.phase == "gf":
        from cscwd_router.code.scripts.train_router_dinov3 import parse_args as base_args
    else:
        from cscwd_router.code.scripts.transfer_gf_router_to_vat import parse_args as base_args
    args = base_args(remaining)
    args.condition = selected.condition
    return selected.phase, args


def sources():
    return {name: sha256(ROOT / name) for name in (
        "cscwd_router/code/stag/conditional.py", "cscwd_router/code/scripts/train_conditioned_router.py",
        "cscwd_router/code/scripts/run_final_ablation.py")}


def run(phase, args):
    factory = functools.partial(get_conditional_model, condition=args.condition)
    if phase == "gf":
        if args.dataset != "gazefollow" or args.size != "b":
            raise ValueError("the final ablation recipe is GF ViT-B only")
        from cscwd_router.code.stag.dinov3_run import run as train
    else:
        source = json.loads(Path(args.gf_summary).read_text())
        if source["model_config"].get("routing_condition", "full") != args.condition:
            raise ValueError("VAT must inherit its own conditional GF branch")
        from cscwd_router.code.stag.dinov3_transfer import run as train
    return train(args, model_factory=factory, extra_sources=sources())


if __name__ == "__main__":
    run(*parse_args())
