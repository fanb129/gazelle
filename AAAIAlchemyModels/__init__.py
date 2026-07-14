"""Isolated effect-first model candidates for the AAAI 2027 P3 study."""

from .factory import CANDIDATE_CHOICES, build_p3_alchemy_model, resolve_candidate
from .model import AlchemyGazeLLE

__all__ = [
    "AlchemyGazeLLE",
    "CANDIDATE_CHOICES",
    "build_p3_alchemy_model",
    "resolve_candidate",
]
