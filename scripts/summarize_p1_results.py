import argparse
import csv
import json
from pathlib import Path


METRIC_FIELDS = ("auc", "l2", "inout_ap")
FEATURE_METRICS = (
    "crowd_token_cosine",
    "inter_person_boundary_separability",
    "foreground_background_contrast",
)


def format_cell(value):
    if value is None or value == "":
        return "TBD"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{float(value):.4f}"
    if isinstance(value, str):
        if value == "TBD":
            return "TBD"
        try:
            return f"{float(value):.4f}"
        except ValueError:
            return value
    return str(value)


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def collect_metric_rows(input_dir):
    spatial_rows = []
    fusion_rows = []
    for path in sorted(input_dir.rglob("metrics.json")):
        if any(part.startswith("reliability") for part in path.parts):
            continue
        payload = load_json(path)
        rows = payload.get("rows", [])
        for row in rows:
            enriched = dict(row)
            enriched["source_file"] = str(path)
            group = classify_metric_row(path, row)
            if group == "fusion":
                fusion_rows.append(enriched)
            else:
                spatial_rows.append(enriched)
    return spatial_rows, fusion_rows


def classify_metric_row(path, row):
    parts = set(path.parts)
    if "fusion" in parts:
        return "fusion"
    if "spatial_prior" in parts:
        return "spatial_prior"
    if row.get("fusion") in {"raw_concat", "equal_weight", "fpn", "selected_layers"}:
        return "fusion"
    return "spatial_prior"


def collect_feature_rows(input_dir):
    path = input_dir / "feature_oversmoothing.json"
    if not path.exists():
        return []
    payload = load_json(path)
    rows = []
    for row in payload.get("rows", []):
        metrics = row.get("metrics", {})
        rows.append(
            {
                "backbone": row.get("backbone", "TBD"),
                "layer": row.get("layer", "TBD"),
                "layer_index": row.get("layer_index", "TBD"),
                "sample_count": row.get("sample_count", "TBD"),
                "crowd_token_cosine": metric_value(metrics.get("crowd_token_cosine")),
                "inter_person_boundary_separability": metric_value(metrics.get("inter_person_boundary_separability")),
                "foreground_background_contrast": metric_value(metrics.get("foreground_background_contrast")),
            }
        )
    return rows


def metric_value(metric):
    if not metric or not metric.get("computed", False):
        return "TBD"
    return metric.get("value", "TBD")


def collect_reliability_rows(input_dir):
    rows = []
    for path in sorted(input_dir.rglob("aggregate_metrics.json")):
        payload = load_json(path)
        for row in payload.get("rows", []):
            enriched = dict(row)
            enriched["source_file"] = str(path)
            rows.append(enriched)
    return rows


def markdown_table(headers, rows):
    output = []
    output.append("| " + " | ".join(headers) + " |")
    output.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return "\n".join(output)


def metric_row_cells(row):
    return [
        format_cell(row.get("variant")),
        format_cell(row.get("spatial_prior")),
        format_cell(row.get("fusion")),
        format_cell(row.get("seed")),
        format_cell(row.get("status")),
        format_cell(row.get("sample_count")),
        format_cell(row.get("auc")),
        format_cell(row.get("l2")),
        format_cell(row.get("inout_ap")),
    ]


def fusion_row_cells(row):
    return [
        format_cell(row.get("variant")),
        format_cell(row.get("spatial_prior")),
        format_cell(row.get("fusion")),
        format_cell(row.get("selected_layers")),
        format_cell(row.get("seed")),
        format_cell(row.get("status")),
        format_cell(row.get("sample_count")),
        format_cell(row.get("auc")),
        format_cell(row.get("l2")),
        format_cell(row.get("inout_ap")),
    ]


def feature_row_cells(row):
    return [
        format_cell(row.get("backbone")),
        format_cell(row.get("layer")),
        format_cell(row.get("crowd_token_cosine")),
        format_cell(row.get("inter_person_boundary_separability")),
        format_cell(row.get("foreground_background_contrast")),
        format_cell(row.get("sample_count")),
    ]


def reliability_row_cells(row):
    return [
        format_cell(row.get("variant")),
        format_cell(row.get("status")),
        join_values(row.get("seeds", [])),
        join_values(row.get("expected_seeds", [])),
        join_values(row.get("missing_seeds", [])),
        mean_std_cell(row, "auc"),
        mean_std_cell(row, "l2"),
        mean_std_cell(row, "inout_ap"),
    ]


def join_values(values):
    if values is None:
        return "TBD"
    if not values:
        return ""
    return " ".join(str(value) for value in values)


def mean_std_cell(row, metric):
    mean_value = row.get(f"{metric}_mean", "TBD")
    std_value = row.get(f"{metric}_std", "TBD")
    if mean_value == "TBD" or std_value == "TBD":
        return "TBD"
    return f"{format_cell(mean_value)} +/- {format_cell(std_value)}"


def build_markdown(input_dir, spatial_rows, fusion_rows, feature_rows, reliability_rows):
    spatial_display = [metric_row_cells(row) for row in spatial_rows] or template_spatial_rows()
    fusion_display = [fusion_row_cells(row) for row in fusion_rows] or template_fusion_rows()
    feature_display = [feature_row_cells(row) for row in feature_rows] or template_feature_rows()
    reliability_display = [reliability_row_cells(row) for row in reliability_rows] or template_reliability_rows()

    sections = [
        "# P1 Controls and Analysis Tables",
        "",
        "## No Invented Numbers",
        "",
        "No metric cell may be filled without a server output file. Missing, pending, or runtime-limited values remain `TBD`.",
        "",
        f"Input directory: `{input_dir}`",
        "",
        "## Spatial-Prior Controls",
        markdown_table(
            ["Variant", "Spatial Prior", "Fusion", "Seed", "Status", "Samples", "AUC", "L2", "In/Out AP"],
            spatial_display,
        ),
        "",
        "## Fusion Controls",
        markdown_table(
            ["Variant", "Spatial Prior", "Fusion", "Selected Layers", "Seed", "Status", "Samples", "AUC", "L2", "In/Out AP"],
            fusion_display,
        ),
        "",
        "## Feature Over-Smoothing Metrics",
        markdown_table(
            ["Backbone", "Layer", "Crowd Token Cosine", "Boundary Separability", "Foreground/Background Contrast", "Samples"],
            feature_display,
        ),
        "",
        "## Reliability Mean/Std",
        markdown_table(
            ["Method", "Status", "Seeds", "Expected Seeds", "Missing Seeds", "AUC mean/std", "L2 mean/std", "In/Out AP mean/std"],
            reliability_display,
        ),
        "",
    ]
    return "\n".join(sections)


def template_spatial_rows():
    variants = [
        ("none", "none", "sasa", "required"),
        ("fixed_gaussian", "fixed_gaussian", "sasa", "required"),
        ("coordconv", "coordconv", "sasa", "required"),
        ("ggsf", "ggsf", "sasa", "required"),
        ("fixed_sector", "fixed_sector", "sasa", "optional"),
    ]
    return [[variant, prior, fusion, "3106", status, "TBD", "TBD", "TBD", "TBD"] for variant, prior, fusion, status in variants]


def template_fusion_rows():
    variants = [
        ("raw_concat", "ggsf", "raw_concat", "all", "required"),
        ("equal_weight", "ggsf", "equal_weight", "all", "required"),
        ("fpn", "ggsf", "fpn", "all", "required"),
        ("sasa", "ggsf", "sasa", "all", "required"),
        ("selected_layers", "ggsf", "selected_layers", "shallow/mid/deep", "optional"),
    ]
    return [[variant, prior, fusion, layers, "3106", status, "TBD", "TBD", "TBD", "TBD"] for variant, prior, fusion, layers, status in variants]


def template_feature_rows():
    rows = []
    for backbone in ("dinov2_vitb16", "dinov3_vitb16"):
        for layer in ("shallow", "mid", "deep", "last"):
            rows.append([backbone, layer, "TBD", "TBD", "TBD", "TBD"])
    return rows


def template_reliability_rows():
    return [
        ["baseline_full", "TBD", "TBD", "3106 3107 3108", "TBD", "TBD", "TBD", "TBD"],
        ["gazespot_full", "TBD", "TBD", "3106 3107 3108", "TBD", "TBD", "TBD", "TBD"],
    ]


def build_plot_rows(spatial_rows, fusion_rows, feature_rows, reliability_rows):
    rows = []
    for group, source_rows in (("spatial_prior", spatial_rows), ("fusion", fusion_rows)):
        for row in source_rows:
            for metric in METRIC_FIELDS:
                rows.append(
                    {
                        "plot_group": group,
                        "dataset_split": format_cell(row.get("dataset_split")),
                        "variant": format_cell(row.get("variant")),
                        "backbone": "",
                        "layer": "",
                        "metric": metric,
                        "value": format_cell(row.get(metric)),
                        "sample_count": format_cell(row.get("sample_count")),
                        "status": format_cell(row.get("status")),
                        "notes": "",
                    }
                )
    for row in feature_rows:
        for metric in FEATURE_METRICS:
            rows.append(
                {
                    "plot_group": "feature_oversmoothing",
                    "dataset_split": "VAT Crowd >=4",
                    "variant": "",
                    "backbone": format_cell(row.get("backbone")),
                    "layer": format_cell(row.get("layer")),
                    "metric": metric,
                    "value": format_cell(row.get(metric)),
                    "sample_count": format_cell(row.get("sample_count")),
                    "status": "computed" if row.get(metric) != "TBD" else "incomplete",
                    "notes": "",
                }
            )
    for row in reliability_rows:
        for metric in ("auc", "l2", "inout_ap"):
            rows.append(
                {
                    "plot_group": "reliability",
                    "dataset_split": format_cell(row.get("dataset_split")),
                    "variant": format_cell(row.get("variant")),
                    "backbone": "",
                    "layer": "",
                    "metric": f"{metric}_mean",
                    "value": format_cell(row.get(f"{metric}_mean")),
                    "sample_count": "",
                    "status": format_cell(row.get("status")),
                    "notes": format_cell(row.get("notes")),
                }
            )
    return rows


def write_plot_csv(path, rows):
    fieldnames = ["plot_group", "dataset_split", "variant", "backbone", "layer", "metric", "value", "sample_count", "status", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize P1 rebuttal JSON outputs into Markdown tables and plot CSV.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", "--plot_csv", dest="plot_csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_md = Path(args.output_md)
    plot_csv = Path(args.plot_csv)

    spatial_rows, fusion_rows = collect_metric_rows(input_dir)
    feature_rows = collect_feature_rows(input_dir)
    reliability_rows = collect_reliability_rows(input_dir)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown(input_dir, spatial_rows, fusion_rows, feature_rows, reliability_rows) + "\n")
    write_plot_csv(plot_csv, build_plot_rows(spatial_rows, fusion_rows, feature_rows, reliability_rows))
    print(f"Wrote P1 Markdown tables: {output_md}")
    print(f"Wrote P1 plot data CSV: {plot_csv}")


if __name__ == "__main__":
    main()
