import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gazelle.ablation_variants import (
    RunMetadata,
    parse_selected_layers,
    resolve_variant_config,
)


def test_existing_sasa_ggsf_flags_map_to_production_modes():
    config = resolve_variant_config(use_sasa=True, use_ggsf=True, seed=3106)

    assert config.spatial_prior == "ggsf"
    assert config.fusion == "sasa"
    assert config.use_sasa is True
    assert config.use_ggsf is True


def test_explicit_rebuttal_modes_override_legacy_booleans():
    config = resolve_variant_config(
        spatial_prior="none",
        fusion="raw_concat",
        use_sasa=True,
        use_ggsf=True,
        seed=3107,
    )

    assert config.spatial_prior == "none"
    assert config.fusion == "raw_concat"
    assert config.use_sasa is False
    assert config.use_ggsf is False
    assert config.seed == 3107


def test_selected_layers_accept_named_presets_and_csv_indices():
    assert parse_selected_layers("shallow_mid") == [2, 5]
    assert parse_selected_layers("2,5,8,11") == [2, 5, 8, 11]


def test_run_metadata_serializes_required_p1_fields():
    metadata = RunMetadata(
        dataset="vat",
        dataset_split="VAT Crowd >=4",
        backbone="gazelle_dinov3_vitb16_inout",
        input_size=[512, 512],
        seed=3106,
        spatial_prior="ggsf",
        fusion="sasa",
        selected_layers=[2, 5, 8, 11],
        checkpoint_path="./checkpoints/gazelle_dinov3_vitb16.pt",
        sample_count=None,
    )

    encoded = json.loads(metadata.to_json())

    assert encoded["dataset_split"] == "VAT Crowd >=4"
    assert encoded["backbone"] == "gazelle_dinov3_vitb16_inout"
    assert encoded["input_size"] == [512, 512]
    assert encoded["seed"] == 3106
    assert encoded["spatial_prior"] == "ggsf"
    assert encoded["fusion"] == "sasa"
    assert encoded["selected_layers"] == [2, 5, 8, 11]
    assert encoded["checkpoint_path"] == "./checkpoints/gazelle_dinov3_vitb16.pt"
    assert "sample_count" in encoded


def test_plan_only_runner_writes_resolved_run_plan(tmp_path):
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_rebuttal_ablation.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--print_plan_only",
            "--group",
            "spatial_prior",
            "--dataset",
            "vat",
            "--data_path",
            "/data/vat",
            "--crowd_json",
            "/data/vat/test_preprocessed_subsets/test_crowd_ge4.json",
            "--variants",
            "none",
            "ggsf",
            "--seed",
            "3106",
            "--output_dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "run_plan.json" in result.stdout
    plan = json.loads((tmp_path / "run_plan.json").read_text())
    assert plan["print_plan_only"] is True
    assert [run["metadata"]["spatial_prior"] for run in plan["runs"]] == ["none", "ggsf"]
    assert all(run["metadata"]["fusion"] == "sasa" for run in plan["runs"])
