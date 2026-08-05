import pytest

torch = pytest.importorskip("torch")

from gazelle.routing.backbone import gather_patch_tokens, gather_rope, scatter_patch_tokens


def test_gather_patch_tokens_uses_per_image_indices():
    tokens = torch.arange(2 * 6 * 3).reshape(2, 6, 3)
    indices = torch.tensor([[0, 2, 5], [1, 3, 4]])

    selected = gather_patch_tokens(tokens, indices)

    assert torch.equal(selected[0], tokens[0, indices[0]])
    assert torch.equal(selected[1], tokens[1, indices[1]])


def test_gather_rope_keeps_original_spatial_positions():
    sin = torch.arange(6 * 4).reshape(6, 4)
    cos = sin + 100
    indices = torch.tensor([[0, 3], [2, 5]])

    selected_sin, selected_cos = gather_rope((sin, cos), indices)

    assert selected_sin.shape == (2, 2, 4)
    assert torch.equal(selected_sin[0], sin[indices[0]])
    assert torch.equal(selected_cos[1], cos[indices[1]])


def test_gather_rope_supports_batched_head_dimension():
    sin = torch.arange(2 * 3 * 6 * 4).reshape(2, 3, 6, 4)
    indices = torch.tensor([[0, 3], [2, 5]])

    selected_sin, _ = gather_rope((sin, sin + 1), indices)

    assert selected_sin.shape == (2, 3, 2, 4)
    assert torch.equal(selected_sin[1, :, 1], sin[1, :, 5])


def test_scatter_uses_early_exit_values_for_unselected_tokens():
    base = torch.zeros(2, 6, 3)
    sparse = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    indices = torch.tensor([[1, 4], [0, 5]])

    dense = scatter_patch_tokens(base, sparse, indices)

    assert torch.equal(dense[0, 1], sparse[0, 0])
    assert torch.equal(dense[0, 4], sparse[0, 1])
    assert torch.equal(dense[1, 0], sparse[1, 0])
    assert torch.equal(dense[1, 5], sparse[1, 1])
    assert dense[0, [0, 2, 3, 5]].eq(0).all()


def test_inplace_scatter_helper_matches_out_of_place_values_and_gradients():
    torch.manual_seed(23)
    indices = torch.tensor([[0, 3, 5], [1, 2, 4]])
    base = torch.randn(2, 6, 4, requires_grad=True)
    sparse = torch.randn(2, 3, 4, requires_grad=True)
    reference_base = base.detach().clone().requires_grad_(True)
    reference_sparse = sparse.detach().clone().requires_grad_(True)

    actual = scatter_patch_tokens(base, sparse, indices)
    reference = reference_base.clone().scatter(
        1,
        indices.unsqueeze(-1).expand_as(reference_sparse),
        reference_sparse,
    )
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (reference * weights).sum().backward()

    assert torch.equal(actual, reference)
    assert torch.equal(base.grad, reference_base.grad)
    assert torch.equal(sparse.grad, reference_sparse.grad)
