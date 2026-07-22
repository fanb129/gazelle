import pytest

torch = pytest.importorskip("torch")

from gazelle.routing.losses import coverage_router_loss
from gazelle.routing.router import CoverageAwareSpatialRouter


def test_router_builds_one_shared_fixed_budget_route_per_image():
    torch.manual_seed(7)
    router = CoverageAwareSpatialRouter(
        in_dim=8,
        hidden_dim=16,
        keep_ratio=0.25,
        escape_tokens=0,
    )
    scene = torch.randn(2, 8, 4, 4)
    bboxes = [
        [[0.00, 0.00, 0.20, 0.20], [0.75, 0.75, 0.95, 0.95]],
        [[0.25, 0.25, 0.45, 0.45]],
    ]

    output = router(scene, bboxes)

    assert output.support_probs.shape == (3, 4, 4)
    assert output.person_to_image.tolist() == [0, 0, 1]
    assert output.image_keep_indices.shape == (2, 4)
    assert output.image_hard_masks.flatten(1).sum(dim=1).tolist() == [4, 4]
    assert output.actual_keep_ratio == 0.25
    assert torch.equal(output.image_keep_indices, output.image_keep_indices.sort(dim=1).values)

    expected_union = torch.maximum(output.support_probs[0], output.support_probs[1])
    assert torch.allclose(output.image_union_probs[0], expected_union)
    assert torch.allclose(output.support_probs.flatten(1).sum(dim=1), torch.ones(3))


def test_head_tokens_are_prioritized_without_changing_fixed_budget():
    router = CoverageAwareSpatialRouter(in_dim=4, hidden_dim=8, keep_ratio=0.05, escape_tokens=0)
    output = router(
        torch.randn(1, 4, 4, 4),
        [[[0.0, 0.0, 0.75, 0.75]]],
    )

    # A 3x3 head footprint is larger than the one-token budget. The selected
    # token stays inside the head, but the budget remains exactly one token.
    assert output.image_keep_indices.shape[1] == 1
    assert output.image_hard_masks[0, :3, :3].sum() == 1
    assert output.actual_keep_ratio == 1 / 16


def test_route_is_invariant_to_an_unrelated_batch_companion():
    router = CoverageAwareSpatialRouter(in_dim=4, hidden_dim=8, keep_ratio=0.25, escape_tokens=0)
    first_scene = torch.randn(1, 4, 4, 4)
    first_bbox = [[0.1, 0.1, 0.3, 0.3]]
    route_alone = router(first_scene, [first_bbox]).image_keep_indices[0]
    route_batched = router(
        torch.cat([first_scene, torch.randn(1, 4, 4, 4)], dim=0),
        [first_bbox, [[0.0, 0.0, 1.0, 1.0]]],
    ).image_keep_indices[0]

    assert torch.equal(route_alone, route_batched)


def test_coverage_loss_is_finite_and_trains_soft_support():
    torch.manual_seed(11)
    router = CoverageAwareSpatialRouter(in_dim=8, hidden_dim=16, keep_ratio=0.25, escape_tokens=0)
    output = router(
        torch.randn(2, 8, 4, 4),
        [[[0.0, 0.0, 0.2, 0.2]], [[0.7, 0.0, 0.9, 0.2]]],
    )
    targets = torch.zeros(2, 64, 64)
    targets[0, 32:48, 48:64] = 1.0
    targets[1, 16:32, 0:16] = 1.0

    losses = coverage_router_loss(output, targets)
    losses["total"].backward()

    assert all(torch.isfinite(value) for value in losses.values())
    assert router.score_head[-1].weight.grad is not None
    assert router.score_head[-1].weight.grad.abs().sum() > 0


def test_zero_initialized_score_head_releases_upstream_gradients_after_one_step():
    torch.manual_seed(17)
    router = CoverageAwareSpatialRouter(in_dim=4, hidden_dim=8, keep_ratio=0.25, escape_tokens=0)
    optimizer = torch.optim.SGD(router.parameters(), lr=0.5)
    scene = torch.randn(1, 4, 4, 4)
    targets = torch.zeros(1, 64, 64)
    targets[:, 32:48, 48:64] = 1.0

    for _ in range(2):
        optimizer.zero_grad()
        output = router(scene, [[[0.0, 0.0, 0.2, 0.2]]])
        coverage_router_loss(output, targets, entropy_weight=0.0)["total"].backward()
        optimizer.step()

    gradient = router.scene_projection.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


def test_coverage_loss_ignores_out_of_frame_people():
    router = CoverageAwareSpatialRouter(in_dim=4, hidden_dim=8, keep_ratio=0.5, escape_tokens=0)
    output = router(torch.randn(1, 4, 4, 4), [[[0.1, 0.1, 0.3, 0.3]]])
    losses = coverage_router_loss(
        output,
        torch.zeros(1, 64, 64),
        inout=torch.tensor([0]),
    )

    assert torch.isfinite(losses["total"])
    assert losses["coverage"].item() == 0.0
