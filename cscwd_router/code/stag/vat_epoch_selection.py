"""Compare VAT checkpoints using all three metrics and their Pareto frontier."""

import math

METRICS = ("auc", "l2", "inout_ap")
POLICY = "pareto_equal_relative_regret_v1"


def dominates(left, right):
    differences = [(left[key] - right[key]) * (1 if key == "l2" else -1) for key in METRICS]
    return all(value <= 0 for value in differences) and any(value < 0 for value in differences)


def compare_epochs(rows):
    ideal = {key: (min if key == "l2" else max)(row["metrics"][key] for row in rows) for key in METRICS}
    if any(not math.isfinite(row["metrics"][key]) or row["metrics"][key] <= 0
           for row in rows for key in METRICS):
        raise ValueError("relative-regret comparison requires finite positive VAT metrics")
    compared = []
    for row in rows:
        metrics = row["metrics"]
        regret = {key: ((metrics[key] - ideal[key]) if key == "l2" else (ideal[key] - metrics[key])) / ideal[key]
                  for key in METRICS}
        dominated_by = [other["epoch_one_based"] for other in rows if dominates(other["metrics"], metrics)]
        compared.append({**row, "dominated_by": dominated_by, "relative_regret": regret,
                         "mean_relative_regret": sum(regret.values()) / len(METRICS)})
    frontier = [row for row in compared if not row["dominated_by"]]
    recommended = min(frontier, key=lambda row: (row["mean_relative_regret"], row["epoch_one_based"]))
    return {"policy": POLICY, "metric_weights": dict.fromkeys(METRICS, 1), "ideal": ideal,
            "pareto_epochs": [row["epoch_one_based"] for row in frontier],
            "recommended_epoch_one_based": recommended["epoch_one_based"], "rows": compared}


def render_comparison(comparison, *, title, frame_rate):
    lines = [f"# {title}", "", f"Test frame sampling interval: {frame_rate}; precision: FP32; K50.", "",
             "| Epoch (1-based) | AUC ↑ | L2 ↓ | In/out AP ↑ | Pareto | Mean relative regret (%) ↓ |",
             "|---|---:|---:|---:|---|---:|"]
    for row in comparison["rows"]:
        metrics = row["metrics"]
        lines.append(f"| {row['epoch_one_based']} | {metrics['auc']:.6f} | {metrics['l2']:.6f} | "
                     f"{metrics['inout_ap']:.6f} | {'yes' if not row['dominated_by'] else 'no'} | "
                     f"{100 * row['mean_relative_regret']:.4f} |")
    lines += ["", f"Recommended epoch: {comparison['recommended_epoch_one_based']}.",
              "Equal-weight auxiliary score: mean((best_auc - auc) / best_auc, "
              "(l2 - best_l2) / best_l2, (best_ap - ap) / best_ap).",
              "All metrics in each row come from the same checkpoint and test sampling interval."]
    return "\n".join(lines) + "\n"
