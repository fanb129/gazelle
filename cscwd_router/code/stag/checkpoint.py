"""Compact structured checkpoints: decoder/Router weights, external DINOv2 SHA."""

import hashlib
from pathlib import Path

import torch


FORMAT = "cscwd_router_dinov2_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_state(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model, *, stage, epoch, context, metrics=None, optimizer=None, scheduler=None):
    path = Path(path)
    payload = {
        "format": FORMAT, "model_config": model.config, "stage": stage, "epoch": epoch,
        "checkpoint_role": "fixed_final" if path.name == "final.pt" else "epoch_last",
        "keep_ratio": model.encoder.router.keep_ratio, "decoder": clone_state(model.decoder),
        "router": clone_state(model.encoder.router), "context": context, "metrics": metrics,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != FORMAT:
        raise ValueError("unsupported DINOv2 Router checkpoint format")
    return payload


def restore_checkpoint(model, payload, *, optimizer=None, scheduler=None):
    if payload["model_config"] != model.config:
        raise ValueError("checkpoint architecture does not match the current model")
    model.decoder.load_state_dict(payload["decoder"], strict=True)
    model.encoder.router.load_state_dict(payload["router"], strict=True)
    model.set_stage(payload["stage"])
    model.encoder.router.keep_ratio = payload["keep_ratio"]
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
