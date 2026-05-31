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
    CoordConvSpatialAdapter,
    FixedGaussianSpatialPrior,
    FixedSectorSpatialPrior,
    IdentitySpatialPrior,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Smoke-check P1 spatial-prior controls without datasets.")
    parser.add_argument("--variants", nargs="+", default=["none", "fixed_gaussian", "coordconv", "ggsf", "fixed_sector"])
    parser.add_argument("--feat_h", type=int, default=8)
    parser.add_argument("--feat_w", type=int, default=8)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--output_dir", default="rebuttal/results/p1/spatial_prior")
    return parser


def check_variant(variant, feat_h, feat_w, channels):
    bboxes = [[[0.25, 0.25, 0.5, 0.5]], [[0.6, 0.2, 0.8, 0.5]]]
    device = torch.device("cpu")

    if variant == "none":
        prior = IdentitySpatialPrior(feat_h, feat_w)
        first = prior(bboxes, device)
        second = prior(bboxes, device)
        return _mask_result(variant, first, second, prior.metadata())

    if variant == "fixed_gaussian":
        prior = FixedGaussianSpatialPrior(feat_h, feat_w)
        first = prior(bboxes, device)
        second = prior(bboxes, device)
        return _mask_result(variant, first, second, prior.metadata())

    if variant == "fixed_sector":
        prior = FixedSectorSpatialPrior(feat_h, feat_w)
        first = prior(bboxes, device)
        second = prior(bboxes, device)
        return _mask_result(variant, first, second, prior.metadata())

    if variant == "coordconv":
        adapter = CoordConvSpatialAdapter(channels, feat_h, feat_w)
        features = [torch.zeros(2, channels, feat_h, feat_w), torch.ones(2, channels, feat_h, feat_w)]
        first = adapter(features, bboxes)
        second = adapter(features, bboxes)
        deterministic = all(torch.equal(a, b) for a, b in zip(first, second))
        return {
            "variant": variant,
            "status": "ok",
            "output_shapes": [list(feat.shape) for feat in first],
            "deterministic": deterministic,
            "metadata": adapter.metadata(),
        }

    if variant == "ggsf":
        try:
            from gazelle.model import GeometryGuidedSpatialFocus
        except ModuleNotFoundError as exc:
            return {
                "variant": variant,
                "status": "metadata_only",
                "deterministic": True,
                "metadata": {
                    "spatial_prior": "ggsf",
                    "description": "existing learned GeometryGuidedSpatialFocus module",
                    "has_learned_mask_parameters": True,
                    "smoke_note": f"model import skipped because dependency is unavailable: {exc.name}",
                },
            }
        else:
            torch.manual_seed(3106)
            prior = GeometryGuidedSpatialFocus(feat_h, feat_w)
            prior.eval()
            first = prior(bboxes, device)
            second = prior(bboxes, device)
            return _mask_result(
                variant,
                first,
                second,
                {
                    "spatial_prior": "ggsf",
                    "description": "existing learned GeometryGuidedSpatialFocus module",
                    "has_learned_mask_parameters": True,
                },
            )

    raise ValueError(f"unsupported spatial prior variant: {variant}")


def _mask_result(variant, first, second, metadata):
    return {
        "variant": variant,
        "status": "ok",
        "output_shape": list(first.shape),
        "deterministic": bool(torch.equal(first, second)),
        "min": float(first.min().item()),
        "max": float(first.max().item()),
        "metadata": metadata,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    results = [check_variant(variant, args.feat_h, args.feat_w, args.channels) for variant in args.variants]

    failures = [result for result in results if not result.get("deterministic")]
    if failures:
        raise RuntimeError(f"non-deterministic spatial prior smoke result: {failures}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "section2_spatial_prior_smoke.json"
    payload = {
        "section": "2.spatial_prior_controls",
        "feat_h": args.feat_h,
        "feat_w": args.feat_w,
        "channels": args.channels,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + os.linesep)
    print(f"Wrote spatial-prior smoke results: {output_path}")


if __name__ == "__main__":
    main()
