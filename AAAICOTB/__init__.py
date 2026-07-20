"""Counterfactual Observer--Target Binding experiments.

This package is intentionally isolated from the historical ``gazelle`` and
``AAAIScripts`` code.  It reuses their model implementations read-only while
adding frame-grouped VAT training and binding-specific evaluation.
"""

# Keep package import lightweight: the annotation audit can be inspected on a
# machine without the CUDA/PyTorch training environment.
