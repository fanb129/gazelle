"""The baseline loss, full VAT scoring, and chronological TBPTT rollouts."""

import time

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from gazelle.utils import vat_auc, vat_l2
from cscwd.code.stag.model import detach_state


class VATMetrics:
    def __init__(self):
        self.auc_sum = 0.0
        self.l2_sum = 0.0
        self.inside_count = 0
        self.scores = []
        self.labels = []

    def update(self, heatmap, inout, labels, gaze):
        heatmap = heatmap.detach().cpu()
        scores = inout.detach().cpu().tolist()
        labels = labels.cpu().tolist()
        for prediction, label, xy in zip(heatmap, labels, gaze):
            if label == 1:
                self.auc_sum += float(vat_auc(prediction, *xy))
                self.l2_sum += float(vat_l2(prediction, *xy))
                self.inside_count += 1
        self.scores.extend(scores)
        self.labels.extend(labels)

    def compute(self):
        return {
            "auc": self.auc_sum / self.inside_count,
            "l2": self.l2_sum / self.inside_count,
            "inout_ap": float(average_precision_score(self.labels, self.scores)),
            "person_frames": len(self.labels), "gt_in_frame": self.inside_count,
            "gt_out_of_frame": len(self.labels) - self.inside_count,
        }


def checkpoint_rank(metrics, epoch):
    return metrics["auc"], -metrics["l2"], metrics["inout_ap"], -epoch


def task_loss(heatmap, inout, target, labels, valid):
    """GT enters only this loss (and scoring), never the temporal forward."""
    inside = valid & labels.bool()
    mse = F.mse_loss(heatmap[inside], target[inside]) if inside.any() else heatmap.sum() * 0
    bce = F.binary_cross_entropy(inout[valid], labels[valid])
    return mse + 0.1 * bce, mse, bce


def train_epoch(model, chunks, optimizer, device, gradient_clip_norm=1.0, log_every=100):
    model.train()
    state = None
    mse_sum = bce_sum = 0.0
    person_frames = inside_count = steps = 0
    started = time.monotonic()
    for step, batch in enumerate(chunks, start=1):
        optimizer.zero_grad(set_to_none=True)
        valid = batch["valid"].to(device)
        labels = batch["inout"].to(device)
        target = batch["heatmap_target"].to(device)
        heatmaps, inouts = [], []
        for frame in range(valid.shape[1]):
            heatmap, inout, state = model.forward_step(
                batch["x"][:, frame].to(device), batch["z"][:, frame].to(device),
                state, batch["reset"][:, frame].to(device),
            )
            heatmaps.append(heatmap)
            inouts.append(inout)
        heatmaps = torch.stack(heatmaps, dim=1)
        inouts = torch.stack(inouts, dim=1)
        loss, mse, bce = task_loss(heatmaps, inouts, target, labels, valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), gradient_clip_norm)
        optimizer.step()
        state = detach_state(state)
        n = int(valid.sum())
        n_inside = int((valid & labels.bool()).sum())
        person_frames += n
        inside_count += n_inside
        mse_sum += float(mse.detach()) * n_inside
        bce_sum += float(bce.detach()) * n
        steps = step
        if step % log_every == 0:
            print(f"Train chunk {step}: frames={person_frames} loss={float(loss.detach()):.6f}", flush=True)
    mse = mse_sum / inside_count
    bce = bce_sum / person_frames
    return {
        "loss": mse + 0.1 * bce, "heatmap_mse": mse, "inout_bce": bce,
        "person_frames": person_frames, "gt_in_frame": inside_count,
        "optimizer_steps": steps, "seconds": time.monotonic() - started,
    }


@torch.no_grad()
def evaluate(model, chunks, device, baseline=False, log_every=100):
    model.eval()
    state = None
    metrics = VATMetrics()
    started = time.monotonic()
    for step, batch in enumerate(chunks, start=1):
        for frame in range(batch["valid"].shape[1]):
            valid = batch["valid"][:, frame]
            if not valid.any():
                continue
            x = batch["x"][:, frame].to(device)
            z = batch["z"][:, frame].to(device)
            if baseline:
                heatmap, inout = model.heads(x, z)
            else:
                heatmap, inout, state = model.forward_step(
                    x, z, state, batch["reset"][:, frame].to(device),
                )
            mask = valid.to(device)
            metrics.update(heatmap[mask], inout[mask], batch["inout"][:, frame][valid],
                           batch["gaze"][:, frame][valid.numpy()])
        if step % log_every == 0:
            print(f"Test chunk {step}: frames={len(metrics.labels)}", flush=True)
    result = metrics.compute()
    result["seconds"] = time.monotonic() - started
    return result
