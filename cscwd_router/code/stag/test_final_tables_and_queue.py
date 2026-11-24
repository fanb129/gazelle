"""Synthetic metadata and CPU dispatch checks, never measured paper results."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from cscwd_router.code.scripts import summarize_final_ablation as table, run_final_ablation_queue as queue
from cscwd_router.code.scripts import run_final_ablation as branch


def table_fixture(tmp_path, monkeypatch):
    recipe = dict.fromkeys(("support_epochs", "decoder_epochs", "batch_size", "grad_accumulation", "router_lr", "decoder_lr",
        "inout_lr", "weight_decay", "clip_norm", "budget_weight", "router_hidden_dim", "router_temperature", "escape_tokens", "seed",
        "epochs", "router_warmup_epochs", "router_loss_weight", "train_frame_rate", "test_frame_rate"), 1)
    recipe["seed"] = 0
    loaded = {}
    for condition in ("scene_only", "scene_geometry", "scene_head", "full"):
        gf_checkpoint = {"path": f"/synthetic/{condition}/gf.pt", "sha256": condition}
        for dataset in ("gazefollow", "vat"):
            path = tmp_path / condition / dataset / "vitb/summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"synthetic_fixture": True}))
            config = {"model": "synthetic_b", "keep_ratio": .5}
            if condition != "full":
                config["routing_condition"] = condition
            source = {"model_config": config, "size": "b", "selection_policy": "test_selected",
                "metrics": {"persons": 2, "inframe_persons": 1, "auc": .5, "avg_l2": .2, "l2": .2,
                    "inout_ap": .5, "routing_gt_point_coverage": 1.}, "selection": {"epoch": 0},
                "artifacts": {"best_test_checkpoint": gf_checkpoint}, "training_epochs": {"support": 3, "sparse_decoder": 5},
                "epochs": 8, "router_warmup_epochs": 1, "initialization": {"shared_tensors_identical": True}}
            manifest = {"recipe": deepcopy(recipe), "data": {"test_annotation": {"sha256": dataset}},
                "initialization": {"sha256": "same_dense"}, "gf_checkpoint": gf_checkpoint}
            loaded[path] = source, manifest, {}
    monkeypatch.setattr(table.io, "verified_source", lambda p: loaded[Path(p)])
    args = table.parse_args(["--report-root", str(tmp_path), "--full-gf-summary", str(tmp_path / "full/gazefollow/vitb/summary.json"),
        "--full-vat-summary", str(tmp_path / "full/vat/vitb/summary.json"), "--output", str(tmp_path / "table.json"),
        "--provenance", str(tmp_path / "provenance.md")])
    return args, loaded


def test_table_assembles_all_four_verified_conditions_and_real_source_references(tmp_path, monkeypatch):
    args, loaded = table_fixture(tmp_path, monkeypatch)
    result = table.run(args)
    assert result["all_branches_complete"] and len(result["rows"]) == 4
    assert len(result["request"]["reports"]) == 8
    assert [r["condition"] for r in result["rows"]] == ["scene_only", "scene_geometry", "scene_head", "full"]
    assert "0.500000" in args.output.with_suffix(".md").read_text()


@pytest.mark.parametrize("change,message", [("inheritance", "own GF"), ("recipe", "optimization"), ("condition", "condition differ")])
def test_table_rejects_unmatched_branches(tmp_path, monkeypatch, change, message):
    args, loaded = table_fixture(tmp_path, monkeypatch)
    source, manifest, _ = loaded[tmp_path / "scene_only/vat/vitb/summary.json"]
    if change == "inheritance":
        manifest["gf_checkpoint"] = {"path": "/wrong/full.pt", "sha256": "wrong"}
    elif change == "recipe":
        manifest["recipe"]["decoder_lr"] *= 2
    else:
        source["model_config"]["routing_condition"] = "scene_head"
    with pytest.raises(ValueError, match=message):
        table.run(args)


def test_partial_table_never_fills_missing_results(tmp_path, monkeypatch):
    args, _ = table_fixture(tmp_path, monkeypatch)
    (tmp_path / "scene_head/vat/vitb/summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="not complete"):
        table.run(args)
    args.allow_incomplete = True
    result = table.run(args)
    assert not result["all_branches_complete"]
    assert result["rows"][2] == {"condition": "scene_head", "head_appearance": True, "geometry": False, "completed": False}
    assert "—" in args.output.with_suffix(".md").read_text()


def test_two_gpu_queue_uses_pinned_training_code_and_stops_on_failure(tmp_path, monkeypatch):
    entry = tmp_path / "cscwd_router/code/scripts/run_final_ablation.py"
    entry.parent.mkdir(parents=True)
    entry.touch()
    args, remaining = queue.parse_args(["--training-code-root", str(tmp_path), "--conditions", "scene_only", "scene_head",
        "--gf-data-path", "/gf", "--vat-data-path", "/vat", "--initialization-root", "/init",
        "--artifact-root", "/artifacts", "--report-root", "/reports", "--provenance", "/provenance", "--resume"])
    commands = queue.commands(args, remaining)
    children = [branch.parse_args(command[3:]) for command in commands]
    assert [child.condition for child in children] == ["scene_only", "scene_head"]
    assert all(child.resume and child.seed == 0 for child in children)
    seen = []
    def child(command, **kwargs):
        seen.append(command)
        assert kwargs == {"cwd": tmp_path, "check": True}
        raise RuntimeError("synthetic child failed")
    monkeypatch.setattr(queue.subprocess, "run", child)
    with pytest.raises(RuntimeError, match="synthetic child"):
        queue.run(args, remaining)
    assert seen == commands[:1]
