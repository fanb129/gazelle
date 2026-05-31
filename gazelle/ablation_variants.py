import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Optional


SPATIAL_PRIOR_CHOICES = ("none", "fixed_gaussian", "coordconv", "ggsf", "fixed_sector")
FUSION_CHOICES = ("raw_concat", "equal_weight", "sasa", "fpn", "selected_layers")
DEFAULT_SELECTED_LAYERS = (2, 5, 8, 11)
SELECTED_LAYER_PRESETS = {
    "shallow": (2,),
    "mid": (5, 8),
    "deep": (8,),
    "last": (11,),
    "shallow_mid": (2, 5),
    "mid_deep": (5, 8, 11),
    "all": DEFAULT_SELECTED_LAYERS,
}


@dataclass(frozen=True)
class VariantConfig:
    spatial_prior: str
    fusion: str
    selected_layers: list[int]
    seed: int
    use_sasa: bool
    use_ggsf: bool
    selected_layers_label: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunMetadata:
    dataset: str
    dataset_split: str
    backbone: str
    input_size: list[int]
    seed: int
    spatial_prior: str
    fusion: str
    selected_layers: list[int]
    checkpoint_path: Optional[str]
    sample_count: Optional[int]
    selected_layers_label: Optional[str] = None
    variant: Optional[str] = None
    group: Optional[str] = None
    data_path: Optional[str] = None
    crowd_json: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def add_variant_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--spatial_prior", choices=SPATIAL_PRIOR_CHOICES, default=None)
    parser.add_argument("--fusion", choices=FUSION_CHOICES, default=None)
    parser.add_argument("--selected_layers", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3106)
    return parser


def parse_selected_layers(value: Optional[str]) -> list[int]:
    if value is None or value == "":
        return list(DEFAULT_SELECTED_LAYERS)

    normalized = str(value).strip().lower()
    if normalized in SELECTED_LAYER_PRESETS:
        return list(SELECTED_LAYER_PRESETS[normalized])

    try:
        layers = [int(part.strip()) for part in normalized.split(",") if part.strip()]
    except ValueError as exc:
        choices = ", ".join(sorted(SELECTED_LAYER_PRESETS))
        raise argparse.ArgumentTypeError(
            f"selected_layers must be a comma-separated integer list or one of: {choices}"
        ) from exc

    if not layers:
        raise argparse.ArgumentTypeError("selected_layers cannot be empty")
    if any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("selected_layers must be non-negative indices")
    return layers


def selected_layers_label(value: Optional[str], selected_layers: Iterable[int]) -> Optional[str]:
    if value is None or value == "":
        return "all"

    normalized = str(value).strip().lower()
    if normalized in SELECTED_LAYER_PRESETS:
        return normalized
    return ",".join(str(layer) for layer in selected_layers)


def resolve_variant_config(
    *,
    spatial_prior: Optional[str] = None,
    fusion: Optional[str] = None,
    selected_layers: Optional[str] = None,
    use_sasa: bool = False,
    use_ggsf: bool = False,
    seed: int = 3106,
) -> VariantConfig:
    resolved_spatial = spatial_prior if spatial_prior is not None else ("ggsf" if use_ggsf else "none")
    resolved_fusion = fusion if fusion is not None else ("sasa" if use_sasa else "raw_concat")

    if resolved_spatial not in SPATIAL_PRIOR_CHOICES:
        raise argparse.ArgumentTypeError(f"invalid spatial_prior: {resolved_spatial}")
    if resolved_fusion not in FUSION_CHOICES:
        raise argparse.ArgumentTypeError(f"invalid fusion: {resolved_fusion}")

    parsed_layers = parse_selected_layers(selected_layers)
    return VariantConfig(
        spatial_prior=resolved_spatial,
        fusion=resolved_fusion,
        selected_layers=parsed_layers,
        seed=seed,
        use_sasa=(resolved_fusion == "sasa"),
        use_ggsf=(resolved_spatial == "ggsf"),
        selected_layers_label=selected_layers_label(selected_layers, parsed_layers),
    )


def infer_dataset_split(dataset: str, crowd_json: Optional[str] = None) -> str:
    if dataset == "vat":
        if crowd_json:
            filename = crowd_json.rsplit("/", 1)[-1]
            if "ge4" in filename:
                return "VAT Crowd >=4"
            if "gt4" in filename:
                return "VAT Crowd >4"
            if "eq4" in filename:
                return "VAT Crowd =4"
            return f"VAT Crowd ({filename})"
        return "VAT test"
    if dataset == "gazefollow":
        return "GazeFollow test"
    return dataset


def build_run_metadata(
    *,
    dataset: str,
    backbone: str,
    config: VariantConfig,
    input_size: Iterable[int] = (512, 512),
    checkpoint_path: Optional[str] = None,
    sample_count: Optional[int] = None,
    group: Optional[str] = None,
    variant: Optional[str] = None,
    data_path: Optional[str] = None,
    crowd_json: Optional[str] = None,
) -> RunMetadata:
    return RunMetadata(
        dataset=dataset,
        dataset_split=infer_dataset_split(dataset, crowd_json),
        backbone=backbone,
        input_size=list(input_size),
        seed=config.seed,
        spatial_prior=config.spatial_prior,
        fusion=config.fusion,
        selected_layers=config.selected_layers,
        selected_layers_label=config.selected_layers_label,
        checkpoint_path=checkpoint_path,
        sample_count=sample_count,
        group=group,
        variant=variant,
        data_path=data_path,
        crowd_json=crowd_json,
    )
