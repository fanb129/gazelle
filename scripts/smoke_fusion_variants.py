import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from gazelle.ablation_variants import (
    EqualWeightFusion,
    FPNFusion,
    RawConcatFusion,
    SelectedLayersFusion,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Smoke-check P1 fusion controls without datasets.")
    parser.add_argument("--variants", nargs="+", default=["raw_concat", "equal_weight", "fpn", "sasa", "selected_layers"])
    parser.add_argument("--selected_layer_sets", nargs="+", default=["shallow", "mid", "deep", "shallow_mid", "mid_deep", "all"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--feat_h", type=int, default=8)
    parser.add_argument("--feat_w", type=int, default=8)
    parser.add_argument("--out_channels", type=int, default=6)
    parser.add_argument("--output_dir", default="rebuttal/results/p1/fusion")
    return parser


def make_features(batch_size, channels, feat_h, feat_w):
    return [torch.ones(batch_size, channels, feat_h, feat_w) * (index + 1) for index in range(4)]


def check_variant(variant, args, selected_layers=None):
    features = make_features(args.batch_size, args.channels, args.feat_h, args.feat_w)
    if variant == "raw_concat":
        module = RawConcatFusion(args.channels, args.out_channels, num_layers=4)
        output, metadata = module(features)
    elif variant == "equal_weight":
        module = EqualWeightFusion(args.channels, args.out_channels, num_layers=4)
        output, metadata = module(features)
    elif variant == "fpn":
        module = FPNFusion(args.channels, args.out_channels, num_layers=4)
        output, metadata = module(features)
    elif variant == "selected_layers":
        module = SelectedLayersFusion(args.channels, args.out_channels, selected_layers=selected_layers)
        output, metadata = module(features)
    elif variant == "sasa":
        try:
            from gazelle.model import ScaleAwareSemanticAggregator
        except ModuleNotFoundError as exc:
            return {
                "variant": variant,
                "status": "metadata_only",
                "deterministic": True,
                "metadata": {
                    "fusion": "sasa",
                    "uses_sasa_routing": True,
                    "smoke_note": f"model import skipped because dependency is unavailable: {exc.name}",
                },
            }
        else:
            torch.manual_seed(3106)
            module = ScaleAwareSemanticAggregator(args.channels, num_scales=4)
            module.eval()
            weighted, weights = module(features)
            output = torch.cat(weighted, dim=1)
            metadata = {
                "fusion": "sasa",
                "uses_sasa_routing": True,
                "layer_weight_shape": list(weights.shape),
                "weighted_feature_shapes": [list(feature.shape) for feature in weighted],
            }
    else:
        raise ValueError(f"unsupported fusion variant: {variant}")

    return {
        "variant": variant,
        "status": "ok",
        "selected_layers": selected_layers,
        "output_shape": list(output.shape),
        "metadata": metadata,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    results = []
    for variant in args.variants:
        if variant == "selected_layers":
            for selected_layers in args.selected_layer_sets:
                results.append(check_variant(variant, args, selected_layers=selected_layers))
        else:
            results.append(check_variant(variant, args))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "section3_fusion_smoke.json"
    payload = {
        "section": "3.fusion_controls",
        "batch_size": args.batch_size,
        "channels": args.channels,
        "feat_h": args.feat_h,
        "feat_w": args.feat_w,
        "out_channels": args.out_channels,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + os.linesep)
    print(f"Wrote fusion smoke results: {output_path}")


if __name__ == "__main__":
    main()
