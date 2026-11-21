"""Single-layer DINOv2 features with a dense prefix and a sparse suffix."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from gazelle.backbone_v0_dinov2 import DinoV2Backbone


@dataclass
class PrefixState:
    """Unnormalized block outputs; patch positions are already embedded."""

    tokens: torch.Tensor
    height: int
    width: int
    route_after_block: int


def gather_patch_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select [B, K, C] patches using independent indices for each image."""
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def scatter_patch_tokens(
    early_tokens: torch.Tensor,
    sparse_tokens: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Fill unselected positions with their unchanged route-layer activations."""
    return early_tokens.scatter(1, indices.unsqueeze(-1).expand_as(sparse_tokens), sparse_tokens)


class RoutedDinoV2Backbone(DinoV2Backbone):
    """Keep CLS/special tokens and prune patches after a dense route layer.

    By default, use the official GazeLLE loader and pretrained checkpoint.
    Passing an existing DINOv2 ``model`` reuses its weights without loading a
    second backbone. The state-dict layout and feature-map interface match
    ``DinoV2Backbone``.

    DINOv2 adds its interpolated learned positional embedding exactly once,
    before block 0. Gathering prefix activations also gathers their spatial
    information; the suffix must not add positional embeddings again.
    """

    def __init__(self, model_name: str = "dinov2_vitb14", *, model: Optional[nn.Module] = None):
        if model is None:
            super().__init__(model_name)
        else:
            nn.Module.__init__(self)
            self.model = model
        if self.model.chunked_blocks:
            raise NotImplementedError("sparse routing requires DINOv2 block_chunks=0 (the official hub default)")

    @property
    def prefix_token_count(self) -> int:
        return 1 + self.model.num_register_tokens

    def tokens_to_map(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # Match forward_features: normalize the full sequence, then slice.
        patches = self.model.norm(tokens)[:, self.prefix_token_count :]
        return patches.reshape(tokens.shape[0], height, width, -1).permute(0, 3, 1, 2)

    def forward_prefix(self, images: torch.Tensor, route_after_block: int = 5) -> PrefixState:
        if not 0 <= route_after_block < len(self.model.blocks) - 1:
            raise ValueError(f"route_after_block must be in [0, {len(self.model.blocks) - 2}]")
        height, width = self.get_out_size(images.shape[-2:])
        tokens = self.model.prepare_tokens_with_masks(images)
        for block_index in range(route_after_block + 1):
            tokens = self.model.blocks[block_index](tokens)
        return PrefixState(tokens, height, width, route_after_block)

    def forward_suffix(self, state: PrefixState, keep_indices: torch.Tensor) -> torch.Tensor:
        """Return one dense feature map after running only K selected patches.

        ``keep_indices`` is [B, K], long, unique and sorted in spatial order,
        as produced by the existing Router. Indices exclude special tokens.
        """
        special = state.tokens[:, : self.prefix_token_count]
        early_patches = state.tokens[:, self.prefix_token_count :]
        selected = gather_patch_tokens(early_patches, keep_indices)
        sparse = torch.cat((special, selected), dim=1)
        for block_index in range(state.route_after_block + 1, len(self.model.blocks)):
            sparse = self.model.blocks[block_index](sparse)

        patches = scatter_patch_tokens(early_patches, sparse[:, self.prefix_token_count :], keep_indices)
        dense = torch.cat((sparse[:, : self.prefix_token_count], patches), dim=1)
        return self.tokens_to_map(dense, state.height, state.width)

    def forward(
        self,
        images: torch.Tensor,
        keep_indices: Optional[torch.Tensor] = None,
        route_after_block: int = 5,
    ) -> torch.Tensor:
        """Use the same split/gather/scatter path for sparse and K100 control."""
        state = self.forward_prefix(images, route_after_block)
        if keep_indices is None:
            keep_indices = torch.arange(state.height * state.width, device=images.device).expand(images.shape[0], -1)
        return self.forward_suffix(state, keep_indices)
