#!/usr/bin/env python3
"""Run one P0.5 inference-only intervention on a trained SASA+GGSF model.

The checkpoint and all learned decoder parameters remain unchanged.  This
script only replaces the SASA weights or GGSF mask during forward inference,
then writes the same taxonomy artifacts as ``failure_taxonomy.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AAAIScripts import failure_taxonomy as taxonomy
from AAAIScripts.common import file_manifest


INTERVENTIONS = (
    "original",
    "fixed_mean",
    "shuffle_previous_frame",
    "equal",
    "last_only",
    "shallow_deep",
    "ggsf_identity",
)

FIXED_WEIGHTS = {
    "fixed_mean": (0.0340173, 0.0974448, 0.2007198, 0.6678180),
    "equal": (0.25, 0.25, 0.25, 0.25),
    "last_only": (0.0, 0.0, 0.0, 1.0),
    "shallow_deep": (0.5, 0.0, 0.0, 0.5),
}


def normalized_weights(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (4,):
        raise ValueError(f"expected four layer weights, got shape={vector.shape}")
    if not np.isfinite(vector).all() or (vector < 0).any() or vector.sum() <= 0:
        raise ValueError("layer weights must be finite, non-negative, and have a positive sum")
    return vector / vector.sum()


def intervention_metadata(name: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "name": name,
        "training": False,
        "checkpoint_parameters_changed": False,
    }
    if name in FIXED_WEIGHTS:
        metadata["layer_order"] = ["L2", "L5", "L8", "L11"]
        metadata["layer_weights"] = normalized_weights(FIXED_WEIGHTS[name]).tolist()
    elif name == "shuffle_previous_frame":
        metadata.update({
            "shuffle_unit": "previous frame mean SASA weights",
            "first_frame_fallback": normalized_weights(FIXED_WEIGHTS["fixed_mean"]).tolist(),
            "caveat": (
                "A deterministic one-frame lag breaks image-weight correspondence while preserving "
                "the empirical cross-frame weight distribution up to the first/last boundary. "
                "It does not create inter-person variation when the learned SASA weights are already identical."
            ),
        })
    elif name == "ggsf_identity":
        metadata.update({"ggsf_mask": 1.0, "sasa": "unchanged learned weights"})
    return metadata


def install_intervention(model: Any, name: str) -> Dict[str, Any]:
    """Patch only this in-memory model instance; no source model file is edited."""
    if name not in INTERVENTIONS:
        raise ValueError(f"unknown intervention: {name}")
    metadata = intervention_metadata(name)
    if name == "original":
        return metadata

    import torch

    if name == "ggsf_identity":
        if not getattr(model, "use_ggsf", False) or not hasattr(model, "ggsf"):
            raise RuntimeError("ggsf_identity requires a checkpoint/model with GGSF enabled")

        def identity_ggsf(bboxes, device):
            people = sum(len(boxes) for boxes in bboxes)
            return torch.ones(
                (people, 1, model.featmap_h, model.featmap_w),
                dtype=torch.float32,
                device=device,
            )

        model.ggsf.forward = identity_ggsf
        return metadata

    if not getattr(model, "use_sasa", False) or not hasattr(model, "sasa"):
        raise RuntimeError(f"{name} requires a checkpoint/model with SASA enabled")

    original_forward = model.sasa.forward
    previous_mean = None
    fallback = normalized_weights(FIXED_WEIGHTS["fixed_mean"])

    def intervene_sasa(features):
        nonlocal previous_mean
        if not features or len(features) != 4:
            raise RuntimeError(f"P0.5 expects four feature levels, got {len(features)}")
        people = features[0].shape[0]
        if name == "shuffle_previous_frame":
            _, learned_weights = original_forward(features)
            current_mean = learned_weights.detach().mean(dim=0)
            selected = (
                torch.as_tensor(fallback, device=features[0].device, dtype=features[0].dtype)
                if previous_mean is None
                else previous_mean.to(device=features[0].device, dtype=features[0].dtype)
            )
            previous_mean = current_mean.cpu()
        else:
            selected = torch.as_tensor(
                normalized_weights(FIXED_WEIGHTS[name]),
                device=features[0].device,
                dtype=features[0].dtype,
            )
        weights = selected.unsqueeze(0).expand(people, -1)
        weighted = [feature * weights[:, index, None, None, None] for index, feature in enumerate(features)]
        return weighted, weights

    model.sasa.forward = intervene_sasa
    return metadata


def write_outputs(args, records, frames, intervention, checkpoint_load) -> None:
    summary = taxonomy.summarize(records, frames)
    prefix = Path(args.output_prefix)
    taxonomy.write_csv(prefix.with_suffix(".records.csv"), records, taxonomy.PERSON_FIELDS)
    taxonomy.write_csv(prefix.with_suffix(".frames.csv"), frames, taxonomy.FRAME_FIELDS)
    summary_fields = sorted({key for row in summary for key in row})
    taxonomy.write_csv(prefix.with_suffix(".summary.csv"), summary, summary_fields)
    report = {
        "status": "measured",
        "analysis": "p05_inference_intervention",
        "dataset": args.dataset,
        "model_label": args.model_label,
        "model_source": "current",
        "model": args.model,
        "checkpoint": file_manifest(Path(args.checkpoint)),
        "checkpoint_load": checkpoint_load,
        "annotation": file_manifest(Path(args.json_path)),
        "num_person_records": len(records),
        "num_frame_records": len(frames),
        "intervention": intervention,
        "bin_config": taxonomy.BinConfig(
            shared_target_radius=args.shared_target_radius,
            collapse_cosine=args.collapse_cosine,
        ).as_dict(),
        "metric_notes": {
            "target_confusion": "Post-hoc X-TCR; it does not imply inter-person binding.",
            "association_margin": "distance(pred, nearest wrong cluster) - distance(pred, own cluster); larger is better",
            "causal_scope": "Inference intervention on one trained checkpoint; it diagnoses mechanism necessity but is not a matched retraining comparison.",
        },
        "summary": summary,
    }
    report_path = prefix.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Wrote P0.5 intervention={args.intervention}: {len(records)} person records, {len(frames)} frames")


def run_self_test() -> None:
    for name, expected in FIXED_WEIGHTS.items():
        vector = normalized_weights(expected)
        assert vector.shape == (4,) and np.isclose(vector.sum(), 1.0) and (vector >= 0).all(), name
    assert np.argmax(normalized_weights(FIXED_WEIGHTS["fixed_mean"])) == 3
    assert normalized_weights(FIXED_WEIGHTS["last_only"]).tolist() == [0.0, 0.0, 0.0, 1.0]
    assert intervention_metadata("shuffle_previous_frame")["training"] is False
    print("Self-test passed: seven interventions and all fixed layer weights are valid.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=("vat",), default="vat")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--json-path", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", default="gazelle_dinov3_vitb16_inout")
    parser.add_argument("--model-label")
    parser.add_argument("--intervention", choices=INTERVENTIONS)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--shared-target-radius", type=float, default=0.06)
    parser.add_argument("--collapse-cosine", type=float, default=0.95)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    required = ("data_path", "json_path", "checkpoint", "intervention", "output_prefix")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("normal execution requires: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    if args.model_label is None:
        args.model_label = f"p05_{args.intervention}"
    # Attributes consumed by failure_taxonomy.evaluate/load_model.
    args.model_source = "current"
    args.spatial_prior = "ggsf"
    args.fusion = "sasa"
    args.selected_layers = None
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    model, transform, checkpoint_load = taxonomy.load_model(args)
    intervention = install_intervention(model, args.intervention)
    records, frames = taxonomy.evaluate(args, loaded_model=(model, transform, checkpoint_load))
    write_outputs(args, records, frames, intervention, checkpoint_load)


if __name__ == "__main__":
    main()
