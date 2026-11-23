"""Read an evaluated Router checkpoint, including an explicitly closed GF run."""

import json
from pathlib import Path

import torch

from cscwd_router.code.stag.checkpoint import sha256


def load_source(summary_path, *, allow_closed_gf=False):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(summary_path.with_name("manifest.json").read_text())
    closed = not summary["completed"]
    if closed and not (allow_closed_gf and summary.get("closed_by_user")
                       and summary.get("checkpoint_export_complete")
                       and summary["dataset"] == "gazefollow"
                       and summary["format"] == "cscwd_router_dinov3_interrupted_v1"):
        raise ValueError("source is unfinished; a verified user-closed GF source needs --allow-closed-gf")
    if (manifest["fingerprint"] != summary["context_fingerprint"]
            or summary["model_config"]["keep_ratio"] != 0.5
            or summary["model_config"]["router_stage"] != "backbone_sparse"):
        raise ValueError("source manifest, fingerprint or K50 Router configuration differs")
    for role in ("best_test_checkpoint", "backbone"):
        item = summary["artifacts"][role]
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"source checkpoint or backbone SHA256 mismatch: {role}")
    payload = torch.load(summary["artifacts"]["best_test_checkpoint"]["path"], map_location="cpu", weights_only=False)
    if (payload["context_fingerprint"] != summary["context_fingerprint"]
            or payload["selection"] != summary["selection"]
            or payload["model_config"] != summary["model_config"]):
        raise ValueError("source checkpoint and summary disagree")
    if closed:
        item = summary["artifacts"]["decoder_resume_checkpoint"]
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("closed source resume SHA256 mismatch")
        last = torch.load(item["path"], map_location="cpu", weights_only=False)
        history = json.loads(summary_path.with_name("sparse_decoder_history.json").read_text())
        count = summary["training_epochs"]["sparse_decoder"]
        selected = min(history, key=lambda row: (row["test"]["min_l2"], -row["test"]["auc"], row["epoch"]))
        if (history != last["history"] or [row["epoch"] for row in history] != list(range(count))
                or last["epoch"] + 1 != count or selected["epoch"] != summary["selection"]["epoch"]
                or selected["test"] != summary["selection"]["metrics"]
                or summary["metrics"] != selected["test"]
                or last["context_fingerprint"] != summary["context_fingerprint"]
                or last["selection"] != summary["selection"]
                or payload["epoch"] != selected["epoch"]):
            raise ValueError("closed GF history, actual epochs or selection disagree")
        best = last["best_model_state"]
        if set(best) != set(payload["model_state"]) or any(
                not torch.equal(value, payload["model_state"][name]) for name, value in best.items()):
            raise ValueError("closed GF exported weights differ from the embedded best")
    return summary, manifest, payload
