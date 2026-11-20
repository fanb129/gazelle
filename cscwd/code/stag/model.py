"""Causal residual ConvLSTM over frozen GazeSpot task features."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def detach_state(state):
    """Carry the numerical memory across chunks without the previous graph."""
    return None if state is None else tuple(value.detach() for value in state)


class ConvLSTMCell(nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(2 * hidden_channels, 4 * hidden_channels, 3, padding=1)

    def forward(self, inputs, state=None, reset=None):
        if state is None:
            shape = (inputs.shape[0], self.hidden_channels, *inputs.shape[-2:])
            hidden, cell = inputs.new_zeros(shape), inputs.new_zeros(shape)
        else:
            hidden, cell = state
        if reset is not None:
            keep = (~reset.bool()).to(inputs.dtype).view(-1, 1, 1, 1)
            hidden, cell = hidden * keep, cell * keep
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat((inputs, hidden), dim=1)
        ).chunk(4, dim=1)
        cell = forget_gate.sigmoid() * cell + input_gate.sigmoid() * candidate.tanh()
        hidden = output_gate.sigmoid() * cell.tanh()
        return hidden, cell


class TemporalAdapter(nn.Module):
    def __init__(self, hidden_channels=64, feature_channels=256):
        super().__init__()
        self.spatial_input = nn.Conv2d(feature_channels, hidden_channels, 1)
        self.token_input = nn.Linear(feature_channels, hidden_channels)
        self.cell = ConvLSTMCell(hidden_channels)
        self.spatial_output = nn.Conv2d(hidden_channels, feature_channels, 1)
        self.token_output = nn.Linear(hidden_channels, feature_channels)
        for projection in (self.spatial_output, self.token_output):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward_step(self, x, z, state=None, reset=None):
        inputs = self.spatial_input(x) + self.token_input(z)[:, :, None, None]
        hidden, cell = self.cell(inputs, state, reset)
        x_prime = x + self.spatial_output(hidden)
        z_prime = z + self.token_output(hidden.mean(dim=(-2, -1)))
        return x_prime, z_prime, (hidden, cell)


class FrozenTaskHeads(nn.Module):
    """Original prediction heads: fixed weights, differentiable inputs.

    The default module definitions reproduce gazelle.model.GazeLLE exactly.
    Supplying the base model's heads also supports extraction-time equivalence
    checks without constructing another backbone.
    """

    def __init__(self, heatmap_head=None, inout_head=None, feature_channels=256,
                 out_size=(64, 64)):
        super().__init__()
        self.out_size = tuple(out_size)
        self.heatmap_head = heatmap_head if heatmap_head is not None else nn.Sequential(
            nn.ConvTranspose2d(feature_channels, feature_channels, 2, stride=2),
            nn.Conv2d(feature_channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.inout_head = inout_head if inout_head is not None else nn.Sequential(
            nn.Linear(feature_channels, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @classmethod
    def from_checkpoint(cls, path: str | Path):
        weights = torch.load(path, map_location="cpu", weights_only=True)
        heads = cls()
        heads.heatmap_head.load_state_dict(weights["heatmap_head"], strict=True)
        heads.inout_head.load_state_dict(weights["inout_head"], strict=True)
        return heads

    def forward(self, x, z):
        # Do not use no_grad here: adapter gradients must pass through the heads.
        heatmap = self.heatmap_head(x).squeeze(1)
        if heatmap.shape[-2:] != self.out_size:
            heatmap = F.interpolate(
                heatmap.unsqueeze(1), size=self.out_size, mode="bilinear",
                align_corners=False, antialias=True,
            ).squeeze(1)
        return heatmap, self.inout_head(z).squeeze(-1)


class TemporalModel(nn.Module):
    def __init__(self, heads, hidden_channels=64, feature_channels=256):
        super().__init__()
        self.adapter = TemporalAdapter(hidden_channels, feature_channels)
        self.heads = heads

    def forward_step(self, x, z, state=None, reset=None):
        x_prime, z_prime, state = self.adapter.forward_step(x, z, state, reset)
        heatmap, inout = self.heads(x_prime, z_prime)
        return heatmap, inout, state
