import json
import pathlib
import subprocess
import sys

import pytest

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


def test_runner_smoke_writes_manifest_and_csv_without_training(tmp_path):
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_rebuttal_ablation.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--runner_smoke_only",
            "--group",
            "fusion",
            "--dataset",
            "vat",
            "--data_path",
            "/data/vat",
            "--crowd_json",
            "/data/vat/test_preprocessed_subsets/test_crowd_gt4.json",
            "--init_ckpt",
            "./checkpoints/gazelle_dinov3_vitb16.pt",
            "--variants",
            "raw_concat",
            "sasa",
            "--seed",
            "3106",
            "--max_epochs",
            "1",
            "--batch_size",
            "2",
            "--output_dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    csv_text = (tmp_path / "metrics.csv").read_text()

    assert manifest["runner_smoke_only"] is True
    assert len(manifest["runs"]) == 2
    assert manifest["runs"][0]["commands"]["train"][0] == sys.executable
    assert "--fusion" in manifest["runs"][0]["commands"]["train"]
    assert manifest["runs"][0]["metadata"]["dataset_split"] == "VAT Crowd >4"
    assert all(row["status"] == "smoke_only" for row in metrics["rows"])
    assert "dataset_split,variant,spatial_prior,fusion,seed,status" in csv_text


def test_reliability_plan_expands_variants_across_seeds(tmp_path):
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_rebuttal_ablation.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--print_plan_only",
            "--group",
            "reliability",
            "--dataset",
            "vat",
            "--data_path",
            "/data/vat",
            "--crowd_json",
            "/data/vat/test_preprocessed_subsets/test_crowd_ge4.json",
            "--variants",
            "baseline_full",
            "gazespot_full",
            "--seeds",
            "3106",
            "3107",
            "3108",
            "--output_dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    plan = json.loads((tmp_path / "run_plan.json").read_text())
    names = [run["name"] for run in plan["runs"]]

    assert plan["section"] == "6.statistical_reliability"
    assert plan["expected_seeds"] == [3106, 3107, 3108]
    assert names == [
        "reliability_baseline_full_seed3106",
        "reliability_gazespot_full_seed3106",
        "reliability_baseline_full_seed3107",
        "reliability_gazespot_full_seed3107",
        "reliability_baseline_full_seed3108",
        "reliability_gazespot_full_seed3108",
    ]
    assert plan["runs"][0]["metadata"]["spatial_prior"] == "none"
    assert plan["runs"][0]["metadata"]["fusion"] == "raw_concat"
    assert plan["runs"][1]["metadata"]["spatial_prior"] == "ggsf"
    assert plan["runs"][1]["metadata"]["fusion"] == "sasa"


def test_reliability_aggregate_reports_mean_std_for_complete_seeds():
    from scripts.run_rebuttal_ablation import aggregate_reliability_rows

    rows = []
    for seed, auc, l2, ap in [(3106, 0.6, 0.3, 0.7), (3107, 0.8, 0.5, 0.9), (3108, 1.0, 0.7, 1.0)]:
        rows.append(
            {
                "dataset_split": "VAT Crowd >=4",
                "variant": "baseline_full",
                "seed": seed,
                "checkpoint_path": f"run-{seed}.pt",
                "sample_count": 8,
                "auc": auc,
                "l2": l2,
                "inout_ap": ap,
                "status": "evaluated",
            }
        )

    aggregate = aggregate_reliability_rows(rows, variants=["baseline_full"], expected_seeds=[3106, 3107, 3108])
    summary = aggregate["rows"][0]

    assert aggregate["section"] == "6.statistical_reliability"
    assert summary["variant"] == "baseline_full"
    assert summary["status"] == "complete"
    assert summary["seeds"] == [3106, 3107, 3108]
    assert summary["missing_seeds"] == []
    assert summary["auc_mean"] == pytest.approx(0.8)
    assert summary["auc_std"] == pytest.approx(0.2)
    assert summary["l2_mean"] == pytest.approx(0.5)
    assert summary["l2_std"] == pytest.approx(0.2)
    assert summary["inout_ap_mean"] == pytest.approx(0.8666666667)
    assert summary["inout_ap_std"] == pytest.approx(0.1527525232)


def test_reliability_aggregate_marks_missing_seed_incomplete():
    from scripts.run_rebuttal_ablation import aggregate_reliability_rows

    rows = [
        {
            "dataset_split": "VAT Crowd >=4",
            "variant": "gazespot_full",
            "seed": 3106,
            "checkpoint_path": "run-3106.pt",
            "sample_count": 8,
            "auc": 0.9,
            "l2": 0.2,
            "inout_ap": 0.95,
            "status": "evaluated",
        }
    ]

    aggregate = aggregate_reliability_rows(rows, variants=["gazespot_full"], expected_seeds=[3106, 3107, 3108])
    summary = aggregate["rows"][0]

    assert summary["status"] == "incomplete"
    assert summary["seeds"] == [3106]
    assert summary["missing_seeds"] == [3107, 3108]
    assert summary["auc_mean"] == "TBD"
    assert summary["l2_std"] == "TBD"
    assert summary["notes"] == "missing seeds: 3107, 3108"


def test_runner_fails_with_available_crowd_json_alternatives(tmp_path):
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_rebuttal_ablation.py"
    subset_dir = tmp_path / "test_preprocessed_subsets"
    subset_dir.mkdir()
    (subset_dir / "test_crowd_gt4.json").write_text("[]")

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
            str(tmp_path),
            "--crowd_json",
            str(subset_dir / "test_crowd_ge4.json"),
            "--variants",
            "none",
            "--output_dir",
            str(tmp_path / "out"),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "test_crowd_gt4.json" in result.stderr


def test_identity_spatial_prior_returns_all_ones_gate():
    torch = pytest.importorskip("torch")
    from gazelle.ablation_variants import IdentitySpatialPrior

    prior = IdentitySpatialPrior(feat_h=4, feat_w=5)

    mask = prior([[[0.2, 0.2, 0.4, 0.4]], [[0.1, 0.1, 0.3, 0.3]]], torch.device("cpu"))

    assert mask.shape == (2, 1, 4, 5)
    assert torch.all(mask == 1)
    assert prior.metadata()["spatial_prior"] == "none"


def test_fixed_gaussian_prior_is_deterministic_and_documents_sigma_rule():
    torch = pytest.importorskip("torch")
    from gazelle.ablation_variants import FixedGaussianSpatialPrior

    prior = FixedGaussianSpatialPrior(feat_h=6, feat_w=6)
    bboxes = [[[0.25, 0.25, 0.5, 0.5]], [[0.6, 0.2, 0.8, 0.5]]]

    mask_a = prior(bboxes, torch.device("cpu"))
    mask_b = prior(bboxes, torch.device("cpu"))

    assert mask_a.shape == (2, 1, 6, 6)
    assert torch.equal(mask_a, mask_b)
    assert prior.metadata()["sigma_rule"] == "max(head_width, head_height) * 0.75, clamped to one feature cell"


def test_fixed_sector_prior_is_deterministic_and_marked_optional():
    torch = pytest.importorskip("torch")
    from gazelle.ablation_variants import FixedSectorSpatialPrior

    prior = FixedSectorSpatialPrior(feat_h=5, feat_w=5)
    bboxes = [[[0.3, 0.2, 0.5, 0.4]]]

    mask_a = prior(bboxes, torch.device("cpu"))
    mask_b = prior(bboxes, torch.device("cpu"))

    assert mask_a.shape == (1, 1, 5, 5)
    assert torch.equal(mask_a, mask_b)
    assert prior.metadata()["optional"] is True
    assert "without head-pose" in prior.metadata()["limitation"]


def test_coordconv_adapter_preserves_feature_shapes_without_mask_output():
    torch = pytest.importorskip("torch")
    from gazelle.ablation_variants import CoordConvSpatialAdapter

    adapter = CoordConvSpatialAdapter(in_channels=3, feat_h=4, feat_w=4)
    features = [torch.zeros(2, 3, 4, 4), torch.ones(2, 3, 4, 4)]
    bboxes = [[[0.2, 0.2, 0.4, 0.4]], [[0.5, 0.5, 0.7, 0.8]]]

    conditioned = adapter(features, bboxes)

    assert [feat.shape for feat in conditioned] == [feat.shape for feat in features]
    assert adapter.metadata()["conditioning_mode"] == "additive_coordconv"
    assert adapter.metadata()["uses_multiplicative_mask"] is False


def test_raw_concat_fusion_projects_concatenated_layers():
    torch = pytest.importorskip("torch")
    from gazelle.ablation_variants import RawConcatFusion

    features = [torch.ones(2, 3, 4, 4) * i for i in range(4)]
    fusion = RawConcatFusion(in_channels=3, out_channels=5, num_layers=4)

    output, metadata = fusion(features)

    assert output.shape == (2, 5, 4, 4)
    assert metadata["fusion"] == "raw_concat"
    assert metadata["uses_sasa_routing"] is False


def test_equal_weight_fusion_uses_uniform_layer_weights_before_projection():
    torch = pytest.importorskip("torch")
    from gazelle.ablation_variants import EqualWeightFusion

    features = [torch.ones(2, 3, 4, 4) * i for i in range(4)]
    fusion = EqualWeightFusion(in_channels=3, out_channels=5, num_layers=4)

    output, metadata = fusion(features)

    assert output.shape == (2, 5, 4, 4)
    assert metadata["fusion"] == "equal_weight"
    assert metadata["layer_weights"] == [0.25, 0.25, 0.25, 0.25]


def test_fpn_fusion_returns_projected_feature_without_sasa_routing():
    torch = pytest.importorskip("torch")
    from gazelle.ablation_variants import FPNFusion

    features = [torch.ones(2, 3, 4, 4) * i for i in range(4)]
    fusion = FPNFusion(in_channels=3, out_channels=5, num_layers=4)

    output, metadata = fusion(features)

    assert output.shape == (2, 5, 4, 4)
    assert metadata["fusion"] == "fpn"
    assert metadata["uses_sasa_routing"] is False


def test_selected_layers_fusion_uses_named_presets():
    torch = pytest.importorskip("torch")
    from gazelle.ablation_variants import SelectedLayersFusion

    features = [torch.ones(2, 3, 4, 4) * i for i in range(4)]
    fusion = SelectedLayersFusion(in_channels=3, out_channels=5, selected_layers="shallow_mid")

    output, metadata = fusion(features)

    assert output.shape == (2, 5, 4, 4)
    assert metadata["fusion"] == "selected_layers"
    assert metadata["selected_layers"] == [2, 5]
    assert metadata["selected_layer_positions"] == [0, 1]
