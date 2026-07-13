"""AAAI 2027 candidate models kept separate from the historical Gazelle code.

Imports are lazy so the lightweight router can be unit-tested without loading
the complete Gazelle/torchvision evaluation stack.
"""

__all__ = [
    "MODEL_NAMES",
    "PersonConditionedHierarchicalRouter",
    "PersonHierarchicalGazeLLE",
    "build_person_hierarchical_gazelle",
]


def __getattr__(name):
    if name == "PersonConditionedHierarchicalRouter":
        from .person_hierarchical_router import PersonConditionedHierarchicalRouter
        return PersonConditionedHierarchicalRouter
    if name == "PersonHierarchicalGazeLLE":
        from .person_hierarchical_gazelle import PersonHierarchicalGazeLLE
        return PersonHierarchicalGazeLLE
    if name in {"MODEL_NAMES", "build_person_hierarchical_gazelle"}:
        from .factory import MODEL_NAMES, build_person_hierarchical_gazelle
        return MODEL_NAMES if name == "MODEL_NAMES" else build_person_hierarchical_gazelle
    raise AttributeError(name)
