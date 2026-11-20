"""Causal state supervision for stable segments and recent state transitions.

Labels define training losses only. The loss carry is separate from the
model's predicted recurrent state and is never passed into a model step.
"""

import torch
from torch.nn import functional as F

from .temporal_data import iter_track_batches
from .temporal_model import detach_state, joint_loss


SUPERVISION_VERSION = "r4_state_supervision_v1"
SUPERVISION_SETTINGS = {"lambda_time": 0.1, "response_frames": 3}


def state_temporal_loss(scores, labels, valid, reset, carry=None, response_frames=3):
    """Return two masked losses and detached per-slot supervision carry.

    Stable pairs use adjacent valid, non-reset frames with equal GT state.
    Transition BCE covers a GT flip's current frame and up to the next two
    frames. A new flip restarts that window; a reset or padding ends it. The
    current transition mask depends only on the current and previous labels.

    The returned previous score is detached at a chunk boundary, just like
    the model state. Pairs wholly inside the chunk retain both gradients.
    """
    valid, reset = valid.bool(), reset.bool()
    if carry is None:
        previous_score = torch.zeros_like(scores[:, 0])
        previous_label = torch.zeros_like(labels[:, 0])
        previous_valid = torch.zeros_like(valid[:, 0])
        remaining = torch.zeros_like(labels[:, 0], dtype=torch.long)
    else:
        previous_score, previous_label = carry["score"], carry["label"]
        previous_valid, remaining = carry["valid"], carry["remaining"]
    stable_sum, transition_sum = scores.sum() * 0, scores.sum() * 0
    stable_pairs = torch.zeros((), device=scores.device, dtype=torch.long)
    transition_frames = torch.zeros_like(stable_pairs)
    for step in range(scores.shape[1]):
        current_score, current_label = scores[:, step], labels[:, step]
        current_valid = valid[:, step]
        linked = previous_valid & current_valid & ~reset[:, step]
        stable = linked & (current_label == previous_label)
        changed = linked & (current_label != previous_label)
        remaining = torch.where(changed, response_frames,
                                torch.where(linked, remaining, 0))
        transition = current_valid & (remaining > 0)
        stable_sum = stable_sum + (current_score - previous_score).abs()[stable].sum()
        transition_sum = transition_sum + F.binary_cross_entropy(
            current_score[transition], current_label[transition].float(), reduction="sum")
        stable_pairs = stable_pairs + stable.sum()
        transition_frames = transition_frames + transition.sum()
        remaining = torch.where(current_valid, (remaining - 1).clamp_min(0), 0)
        previous_score, previous_label, previous_valid = current_score, current_label, current_valid
    carry = {"score": previous_score.detach(), "label": previous_label.detach(),
             "valid": previous_valid.detach(), "remaining": remaining.detach()}
    losses = {"stable": stable_sum / stable_pairs.clamp_min(1),
              "transition": transition_sum / transition_frames.clamp_min(1),
              "stable_sum": stable_sum, "transition_sum": transition_sum,
              "stable_pairs": stable_pairs, "transition_frames": transition_frames}
    return losses, carry


def _forward_chunk(model, batch, device, state):
    """Keep model observations identical to the original causal trainer."""
    heatmaps, scores = [], []
    for step in range(batch["valid"].shape[1]):
        heatmap, score, state = model.step(
            batch["features"][:, step].to(device=device, dtype=torch.float32),
            batch["raw_heatmap"][:, step].to(device), batch["raw_inout"][:, step].to(device),
            batch["bboxes"][:, step].to(device), state, batch["reset"][:, step].to(device))
        heatmaps.append(heatmap)
        scores.append(score)
    return torch.stack(heatmaps, dim=1), torch.stack(scores, dim=1), state


def train_supervised_epoch(model, optimizer, tracks, arrays, feature_index, features, device,
                           clip_frames, batch_size, seed, lambda_state,
                           lambda_time=0.1, response_frames=3):
    """Train one complete epoch with independent predicted and loss carries.

    Batch objectives use each loss's own eligible population. Epoch summaries
    pool each numerator/count, then reconstruct the same weighted formula;
    they are not an unweighted average of differently sized chunk means.
    """
    model.train()
    state, loss_carry = None, None
    totals = {"heatmap_sum": 0.0, "state_sum": 0.0, "stable_sum": 0.0,
              "transition_sum": 0.0, "valid_frames": 0, "inframe_frames": 0,
              "stable_pairs": 0, "transition_frames": 0, "chunks": 0}
    for batch in iter_track_batches(tracks, arrays, feature_index, features,
                                     clip_frames, batch_size, seed):
        optimizer.zero_grad(set_to_none=True)
        heatmap, score, state = _forward_chunk(model, batch, device, state)
        labels, valid = batch["target_inout"].to(device), batch["valid"].to(device)
        losses = joint_loss(heatmap, score, batch["target_heatmap"].to(device),
                            labels, valid, lambda_state)
        temporal, loss_carry = state_temporal_loss(score, labels, valid,
            batch["reset"].to(device), loss_carry, response_frames)
        total = losses["total"] + lambda_state * lambda_time * (
            temporal["stable"] + temporal["transition"])
        if not torch.isfinite(total):
            raise ValueError("State-supervised training loss is non-finite")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        state = detach_state(state)
        valid_count, inside_count = int(losses["valid_frames"]), int(losses["inframe_frames"])
        totals["heatmap_sum"] += float(losses["heatmap"].detach()) * inside_count
        totals["state_sum"] += float(losses["state"].detach()) * valid_count
        for name in ("stable_sum", "transition_sum"):
            totals[name] += float(temporal[name].detach())
        for name in ("stable_pairs", "transition_frames"):
            totals[name] += int(temporal[name])
        totals["valid_frames"] += valid_count
        totals["inframe_frames"] += inside_count
        totals["chunks"] += 1
        if totals["chunks"] % 100 == 0:
            print(f"State-supervised chunks={totals['chunks']} frames={totals['valid_frames']}", flush=True)
    map_loss = totals["heatmap_sum"] / max(1, totals["inframe_frames"])
    state_loss = totals["state_sum"] / totals["valid_frames"]
    stable_loss = totals["stable_sum"] / max(1, totals["stable_pairs"])
    transition_loss = totals["transition_sum"] / max(1, totals["transition_frames"])
    return {**totals, "heatmap_loss": map_loss, "state_loss": state_loss,
            "stable_loss": stable_loss, "transition_loss": transition_loss,
            "lambda_time": lambda_time, "response_frames": response_frames,
            "total_loss": map_loss + lambda_state * (
                state_loss + lambda_time * (stable_loss + transition_loss))}
