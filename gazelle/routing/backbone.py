from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from gazelle.backbone import DinoV3Backbone


Rope = Optional[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class PrefixState:
    """State produced by the dense prefix of DINOv3."""

    tokens: torch.Tensor
    height: int
    width: int
    rope: Rope
    dense_features: Dict[int, torch.Tensor]
    route_after_block: int


def gather_patch_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather patch tokens with an independent sorted index set per image."""
    if tokens.ndim != 3 or indices.ndim != 2:
        raise ValueError("tokens and indices must have shapes [B, N, C] and [B, K]")
    if tokens.shape[0] != indices.shape[0]:
        raise ValueError("tokens and indices must have the same batch size")
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def gather_rope(rope: Rope, indices: torch.Tensor) -> Rope:
    """Gather RoPE entries corresponding to patch-token indices.

    DINOv3 normally returns RoPE tensors shaped [N, D].  The helper also
    accepts [B, ..., N, D], which makes it useful for tests and future dynamic
    routing variants.
    """
    if rope is None:
        return None
    sin, cos = rope

    def _gather(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2:
            return value[indices]
        if value.shape[0] != indices.shape[0]:
            raise ValueError("batched RoPE and indices must have the same batch size")
        view_shape = [indices.shape[0]] + [1] * (value.ndim - 3) + [indices.shape[1], 1]
        gather_index = indices.view(*view_shape).expand(*value.shape[:-2], indices.shape[1], value.shape[-1])
        return value.gather(-2, gather_index)

    return _gather(sin), _gather(cos)


def scatter_patch_tokens(
    base_tokens: torch.Tensor,
    sparse_tokens: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Scatter sparse deep tokens into a dense early-exit token grid."""
    if base_tokens.ndim != 3 or sparse_tokens.ndim != 3 or indices.ndim != 2:
        raise ValueError("expected [B, N, C], [B, K, C], and [B, K]")
    if sparse_tokens.shape[:2] != indices.shape:
        raise ValueError("sparse token count must match indices")
    if base_tokens.shape[0] != sparse_tokens.shape[0] or base_tokens.shape[2] != sparse_tokens.shape[2]:
        raise ValueError("base and sparse token batch/channel dimensions must match")
    output = base_tokens.clone()
    return output.scatter(1, indices.unsqueeze(-1).expand_as(sparse_tokens), sparse_tokens)


class RoutedDinoV3Backbone(DinoV3Backbone):
    """DINOv3 with a dense prefix and a token-sparse suffix.

    Special tokens are retained.  Patch tokens and their matching RoPE entries
    are pruned together.  Late sparse activations are scattered into a dense
    grid using the route-layer activation as an early-exit fallback.
    """

    @property
    def prefix_token_count(self) -> int:
        return int(self.model.n_storage_tokens) + 1

    def _validate_route(self, route_after_block: int) -> None:
        if getattr(self.model, "chunked_blocks", False):
            raise NotImplementedError("coverage routing currently requires non-chunked DINOv3 blocks")
        if not 0 <= route_after_block < len(self.model.blocks) - 1:
            raise ValueError(
                f"route_after_block must be in [0, {len(self.model.blocks) - 2}], "
                f"got {route_after_block}"
            )

    def tokens_to_map(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Normalize patch tokens and restore their dense spatial layout."""
        patches = self.model.norm(tokens[:, self.prefix_token_count :])
        return patches.reshape(tokens.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()

    def forward_prefix(self, images: torch.Tensor, route_after_block: int) -> PrefixState:
        self._validate_route(route_after_block)
        tokens, (height, width) = self.model.prepare_tokens_with_masks(images)
        rope = self.model.rope_embed(H=height, W=width) if self.model.rope_embed is not None else None
        dense_features: Dict[int, torch.Tensor] = {}

        for block_index in range(route_after_block + 1):
            tokens = self.model.blocks[block_index](tokens, rope)
            if block_index in self.out_indices:
                dense_features[block_index] = self.tokens_to_map(tokens, height, width)

        return PrefixState(
            tokens=tokens,
            height=height,
            width=width,
            rope=rope,
            dense_features=dense_features,
            route_after_block=route_after_block,
        )

    def forward_suffix(self, state: PrefixState, keep_indices: torch.Tensor):
        """Run selected patches through the suffix and return dense feature maps."""
        self._validate_route(state.route_after_block)
        batch_size = state.tokens.shape[0]
        if keep_indices.ndim != 2 or keep_indices.shape[0] != batch_size:
            raise ValueError("keep_indices must have shape [B, K]")
        if keep_indices.dtype != torch.long:
            raise ValueError("keep_indices must use torch.long indices")
        if keep_indices.shape[1] > state.height * state.width:
            raise ValueError("keep_indices cannot contain more entries than the dense patch grid")
        prefix = state.tokens[:, : self.prefix_token_count]
        dense_route_patches = state.tokens[:, self.prefix_token_count :]
        selected_patches = gather_patch_tokens(dense_route_patches, keep_indices)

        # Fixed K keeps the suffix batched, so each block performs one SDPA
        # call instead of a Python loop over samples.
        sparse_tokens = torch.cat([prefix, selected_patches], dim=1)
        if state.rope is None:
            sparse_rope = None
        else:
            selected_rope = gather_rope(state.rope, keep_indices)
            # q/k are [B, heads, K, D_head]; insert a singleton head
            # dimension so per-image RoPE broadcasts over attention heads.
            sin = selected_rope[0].unsqueeze(1) if selected_rope[0].ndim == 3 else selected_rope[0]
            cos = selected_rope[1].unsqueeze(1) if selected_rope[1].ndim == 3 else selected_rope[1]
            sparse_rope = (
                sin,
                cos,
            )

        dense_features = dict(state.dense_features)
        for block_index in range(state.route_after_block + 1, len(self.model.blocks)):
            sparse_tokens = self.model.blocks[block_index](sparse_tokens, sparse_rope)
            if block_index not in self.out_indices:
                continue

            sparse_patches = sparse_tokens[:, self.prefix_token_count :]
            dense_patches = scatter_patch_tokens(dense_route_patches, sparse_patches, keep_indices)
            dense_features[block_index] = self.model.norm(dense_patches).reshape(
                batch_size, state.height, state.width, -1
            ).permute(0, 3, 1, 2).contiguous()

        missing = [index for index in self.out_indices if index not in dense_features]
        if missing:
            raise RuntimeError(
                "route_after_block must not skip requested intermediate layers; "
                f"missing outputs at blocks {missing}"
            )
        return [dense_features[index] for index in self.out_indices]

    def set_trainable_suffix(self, route_after_block: int, trainable: bool = True) -> None:
        """Freeze DINOv3, optionally unfreezing only blocks after the router."""
        self._validate_route(route_after_block)
        for parameter in self.parameters():
            parameter.requires_grad = False
        if trainable:
            for block in self.model.blocks[route_after_block + 1 :]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
