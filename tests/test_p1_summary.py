import csv
import json
import pathlib
import subprocess
import sys


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_summarize_p1_results_generates_tables_and_plot_csv(tmp_path):
    input_dir = tmp_path / "p1"
    output_md = tmp_path / "out" / "p1_tables.md"
    plot_csv = tmp_path / "out" / "p1_plot_data.csv"
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "summarize_p1_results.py"

    write_json(
        input_dir / "spatial_prior" / "metrics.json",
        {
            "rows": [
                {
                    "dataset_split": "VAT Crowd >=4",
                    "variant": "none",
                    "spatial_prior": "none",
                    "fusion": "sasa",
                    "seed": 3106,
                    "status": "evaluated",
                    "sample_count": 8,
                    "auc": 0.5311635739,
                    "l2": 0.5061537823,
                    "inout_ap": "TBD",
                }
            ]
        },
    )
    write_json(
        input_dir / "fusion" / "metrics.json",
        {
            "rows": [
                {
                    "dataset_split": "VAT Crowd >=4",
                    "variant": "sasa",
                    "spatial_prior": "ggsf",
                    "fusion": "sasa",
                    "selected_layers": "all",
                    "seed": 3106,
                    "status": "pending",
                    "sample_count": None,
                    "auc": "TBD",
                    "l2": "TBD",
                    "inout_ap": "TBD",
                }
            ]
        },
    )
    write_json(
        input_dir / "feature_oversmoothing.json",
        {
            "rows": [
                {
                    "backbone": "dinov3_vitb16",
                    "resolved_backbone": "dinov3_vitb16",
                    "layer": "last",
                    "layer_index": 3,
                    "sample_count": 1000,
                    "metrics": {
                        "crowd_token_cosine": {"computed": True, "value": 0.5341709989, "sample_count": 1000},
                        "inter_person_boundary_separability": {"computed": True, "value": 0.207770706, "sample_count": 1000},
                        "foreground_background_contrast": {"computed": True, "value": 0.1673107011, "sample_count": 1000},
                    },
                }
            ]
        },
    )
    write_json(
        input_dir / "reliability" / "aggregate_metrics.json",
        {
            "rows": [
                {
                    "dataset_split": "VAT Crowd >=4",
                    "variant": "gazespot_full",
                    "status": "incomplete",
                    "seeds": [3106],
                    "expected_seeds": [3106, 3107, 3108],
                    "missing_seeds": [3107, 3108],
                    "auc_mean": "TBD",
                    "auc_std": "TBD",
                    "l2_mean": "TBD",
                    "l2_std": "TBD",
                    "inout_ap_mean": "TBD",
                    "inout_ap_std": "TBD",
                    "notes": "missing seeds: 3107, 3108",
                }
            ]
        },
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input_dir",
            str(input_dir),
            "--output_md",
            str(output_md),
            "--plot_csv",
            str(plot_csv),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    markdown = output_md.read_text()
    csv_rows = list(csv.DictReader(plot_csv.open()))

    assert "No metric cell may be filled without a server output file" in markdown
    assert "| none | none | sasa | 3106 | evaluated | 8 | 0.5312 | 0.5062 | TBD |" in markdown
    assert "| sasa | ggsf | sasa | all | 3106 | pending | TBD | TBD | TBD | TBD |" in markdown
    assert "| dinov3_vitb16 | last | 0.5342 | 0.2078 | 0.1673 | 1000 |" in markdown
    assert "| gazespot_full | incomplete | 3106 | 3106 3107 3108 | 3107 3108 | TBD | TBD | TBD |" in markdown
    assert any(row["plot_group"] == "feature_oversmoothing" and row["metric"] == "crowd_token_cosine" for row in csv_rows)
    assert any(row["plot_group"] == "spatial_prior" and row["metric"] == "auc" and row["value"] == "0.5312" for row in csv_rows)
    assert any(row["plot_group"] == "reliability" and row["value"] == "TBD" for row in csv_rows)


def test_p1_controls_document_contains_required_commands_and_templates():
    doc_path = pathlib.Path(__file__).resolve().parents[1] / "rebuttal" / "p1_controls_and_analysis.md"

    text = doc_path.read_text()

    assert "No Invented Numbers" in text
    assert "scripts/run_rebuttal_ablation.py" in text
    assert "scripts/analyze_feature_oversmoothing.py" in text
    assert "scripts/summarize_p1_results.py" in text
    assert "Spatial-Prior Controls" in text
    assert "Feature Over-Smoothing Metrics" in text
    assert "Reliability Mean/Std" in text
