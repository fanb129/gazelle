"""Coverage-aware spatial routing for gaze target estimation."""

from .backbone import RoutedDinoV3Backbone
from .losses import coverage_router_loss
from .model import CoverageAwareGazeLLE, get_coverage_router_model
from .router import CoverageAwareSpatialRouter, RoutingOutput

__all__ = [
    "CoverageAwareGazeLLE",
    "CoverageAwareSpatialRouter",
    "RoutedDinoV3Backbone",
    "RoutingOutput",
    "coverage_router_loss",
    "get_coverage_router_model",
]
