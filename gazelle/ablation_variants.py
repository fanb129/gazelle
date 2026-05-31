import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


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


def _require_torch():
    if torch is None or nn is None:
        raise ModuleNotFoundError("torch is required for spatial-prior modules")


def _module_base():
    return nn.Module if nn is not None else object


class IdentitySpatialPrior(_module_base()):
    def __init__(self, feat_h: int, feat_w: int):
        _require_torch()
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w

    def forward(self, bboxes, device):
        count = _count_bboxes(bboxes)
        return torch.ones((count, 1, self.feat_h, self.feat_w), device=device)

    def metadata(self) -> dict:
        return {
            "spatial_prior": "none",
            "description": "all-ones gate; no spatial filtering",
            "has_learned_mask_parameters": False,
        }


class FixedGaussianSpatialPrior(_module_base()):
    sigma_rule = "max(head_width, head_height) * 0.75, clamped to one feature cell"

    def __init__(self, feat_h: int, feat_w: int, sigma_scale: float = 0.75):
        _require_torch()
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.sigma_scale = sigma_scale

    def forward(self, bboxes, device):
        x_grid, y_grid = _normalized_grid(self.feat_h, self.feat_w, device)
        masks = []
        min_sigma = max(1.0 / self.feat_w, 1.0 / self.feat_h)
        for bbox in _flatten_bboxes(bboxes):
            cx, cy, w, h = _bbox_to_center_size(bbox, device)
            sigma = torch.clamp(torch.maximum(w, h) * self.sigma_scale, min=min_sigma)
            dist_sq = (x_grid - cx).pow(2) + (y_grid - cy).pow(2)
            masks.append(torch.exp(-dist_sq / (2.0 * sigma.pow(2))).unsqueeze(0))
        return torch.stack(masks, dim=0)

    def metadata(self) -> dict:
        return {
            "spatial_prior": "fixed_gaussian",
            "sigma_rule": self.sigma_rule,
            "sigma_scale": self.sigma_scale,
            "has_learned_mask_parameters": False,
        }


class FixedSectorSpatialPrior(_module_base()):
    def __init__(self, feat_h: int, feat_w: int, half_angle_degrees: float = 45.0):
        _require_torch()
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.half_angle_degrees = half_angle_degrees

    def forward(self, bboxes, device):
        x_grid, y_grid = _normalized_grid(self.feat_h, self.feat_w, device)
        cos_threshold = torch.cos(torch.tensor(self.half_angle_degrees * torch.pi / 180.0, device=device))
        masks = []
        for bbox in _flatten_bboxes(bboxes):
            cx, cy, w, h = _bbox_to_center_size(bbox, device)
            dx = x_grid - cx
            dy = y_grid - cy
            radius = torch.sqrt(dx.pow(2) + dy.pow(2)).clamp_min(1e-6)
            downward_cosine = dy / radius
            angular_weight = ((downward_cosine - cos_threshold) / (1.0 - cos_threshold)).clamp(0.0, 1.0)
            sigma = torch.clamp(torch.maximum(w, h) * 2.0, min=max(1.0 / self.feat_w, 1.0 / self.feat_h))
            radial_weight = torch.exp(-radius.pow(2) / (2.0 * sigma.pow(2)))
            masks.append((angular_weight * radial_weight).unsqueeze(0))
        return torch.stack(masks, dim=0)

    def metadata(self) -> dict:
        return {
            "spatial_prior": "fixed_sector",
            "optional": True,
            "half_angle_degrees": self.half_angle_degrees,
            "limitation": "deterministic downward sector without head-pose or gaze-direction cues",
            "has_learned_mask_parameters": False,
        }


class CoordConvSpatialAdapter(_module_base()):
    def __init__(self, in_channels: int, feat_h: int, feat_w: int):
        _require_torch()
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.coord_projection = nn.Conv2d(4, in_channels, kernel_size=1)
        nn.init.zeros_(self.coord_projection.weight)
        nn.init.zeros_(self.coord_projection.bias)

    def forward(self, features_list, bboxes):
        if not features_list:
            return features_list
        device = features_list[0].device
        coord_features = _coord_features(bboxes, self.feat_h, self.feat_w, device)
        coord_projection = self.coord_projection(coord_features)
        return [features + coord_projection.to(dtype=features.dtype) for features in features_list]

    def metadata(self) -> dict:
        return {
            "spatial_prior": "coordconv",
            "conditioning_mode": "additive_coordconv",
            "uses_multiplicative_mask": False,
            "has_learned_mask_parameters": False,
        }


def _count_bboxes(bboxes) -> int:
    return sum(len(bbox_list) for bbox_list in bboxes)


def _flatten_bboxes(bboxes):
    for bbox_list in bboxes:
        for bbox in bbox_list:
            yield bbox


def _normalized_grid(feat_h: int, feat_w: int, device):
    y_grid, x_grid = torch.meshgrid(
        (torch.arange(feat_h, device=device).float() + 0.5) / feat_h,
        (torch.arange(feat_w, device=device).float() + 0.5) / feat_w,
        indexing="ij",
    )
    return x_grid, y_grid


def _bbox_to_center_size(bbox, device):
    if bbox is None:
        values = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
    elif torch.is_tensor(bbox):
        values = bbox.detach().to(device=device, dtype=torch.float32).flatten()
    else:
        values = torch.tensor(list(bbox), device=device, dtype=torch.float32)
    values = values.clamp(0.0, 1.0)
    xmin, ymin, xmax, ymax = values[:4]
    w = (xmax - xmin).clamp_min(1e-6)
    h = (ymax - ymin).clamp_min(1e-6)
    cx = ((xmin + xmax) / 2.0).clamp(0.0, 1.0)
    cy = ((ymin + ymax) / 2.0).clamp(0.0, 1.0)
    return cx, cy, w, h


def _coord_features(bboxes, feat_h: int, feat_w: int, device):
    x_grid, y_grid = _normalized_grid(feat_h, feat_w, device)
    coords = []
    for bbox in _flatten_bboxes(bboxes):
        cx, cy, w, h = _bbox_to_center_size(bbox, device)
        coords.append(
            torch.stack(
                [
                    x_grid - cx,
                    y_grid - cy,
                    torch.full_like(x_grid, w),
                    torch.full_like(y_grid, h),
                ],
                dim=0,
            )
        )
    return torch.stack(coords, dim=0)
