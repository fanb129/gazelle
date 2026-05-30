import numpy as np
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gazelle import eval_utils


def test_zero_jitter_returns_original_normalized_bbox():
    bbox = [0.2, 0.3, 0.4, 0.6]

    assert eval_utils.perturb_normalized_bbox(bbox, 0, np.random.default_rng(123)) == bbox


def test_bbox_perturbation_is_seeded_and_valid():
    bbox = [0.02, 0.03, 0.12, 0.18]
    rng_a = np.random.default_rng(3106)
    rng_b = np.random.default_rng(3106)

    perturbed_a = eval_utils.perturb_normalized_bbox(bbox, 20, rng_a)
    perturbed_b = eval_utils.perturb_normalized_bbox(bbox, 20, rng_b)

    assert perturbed_a == perturbed_b
    assert 0 <= perturbed_a[0] < perturbed_a[2] <= 1
    assert 0 <= perturbed_a[1] < perturbed_a[3] <= 1
    assert perturbed_a != bbox
