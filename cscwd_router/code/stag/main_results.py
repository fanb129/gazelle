"""T4 comparison and the predeclared accuracy/latency decision rule."""

import math


VARIANTS = ("dense", "k100", "k50")
ORDERS = {"forward": VARIANTS, "reverse": tuple(reversed(VARIANTS))}
BATCHES = (1, 4, 8)
AUC_DROP_LIMIT = 0.002
B8_SPEEDUP_MIN = 1.10


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize_results(accuracy, benchmarks):
    official, sparse, matched = (accuracy[name] for name in ("dense_official", "sparse_decoder", "matched_dense"))
    for item in accuracy.values():
        if (item["split"] != "official_test" or item["precision"] != "fp32" or item["batch_size"] != official["batch_size"]
                or item["data"]["annotation_sha256"] != official["data"]["annotation_sha256"]
                or item["training_signature"] != official["training_signature"]
                or any(item["metrics"][key] != official["metrics"][key] for key in ("images", "persons"))):
            raise ValueError("accuracy results must use the same official test, training run and query counts")
    cells = {(item["batch_size"], item["order"]): item for item in benchmarks}
    if len(benchmarks) != 6 or set(cells) != {(batch, order) for batch in BATCHES for order in ORDERS}:
        raise ValueError("T4 requires B1/B4/B8 in both forward and reverse order")
    first = benchmarks[0]
    if (first["precision"] != "fp16" or first["settings"]["autocast_cache_enabled"]
            or not first["settings"]["inference_mode"] or first["settings"]["transfers_and_preprocessing_timed"]
            or first["settings"]["image_size"] != 448 or first["settings"]["people_per_image"] != 1):
        raise ValueError("T4 efficiency requires the declared FP16 inference and input protocol")
    for item in benchmarks:
        if (item["settings"] != first["settings"] or item["environment"]["gpu_uuid"] != first["environment"]["gpu_uuid"]
                or item["precision"] != first["precision"] or item["sources"] != first["sources"]):
            raise ValueError("efficiency cells must share timing settings, GPU, precision and source code")
        for stage, result in (("dense_official", official), ("sparse_decoder", sparse)):
            if item["checkpoints"][stage]["sha256"] != result["checkpoint"]["sha256"]:
                raise ValueError("efficiency must use the same final checkpoints as accuracy")
    efficiency = {}
    for batch in BATCHES:
        forward, reverse = (cells[(batch, order)] for order in ORDERS)
        if forward["input"]["fingerprint"] != reverse["input"]["fingerprint"]:
            raise ValueError("forward/reverse benchmarks must use identical RGB and real head boxes")
        combined = {}
        for variant in VARIANTS:
            timings = [cell["results"][variant]["timing"] for cell in (forward, reverse)]
            combined[variant] = {
                "latency_ms_geomean": geometric_mean([value["latency_ms_median"] for value in timings]),
                "throughput_images_per_second_geomean": geometric_mean([value["throughput_images_per_second_median"] for value in timings]),
                "peak_allocated_mib_max": max(value["peak_allocated_mib"] for value in timings),
            }
        speedups = [cell["results"]["dense"]["timing"]["latency_ms_median"] /
                    cell["results"]["k50"]["timing"]["latency_ms_median"] for cell in (forward, reverse)]
        efficiency[str(batch)] = {
            "variants": combined, "k50_speedup_vs_official_dense": geometric_mean(speedups),
            "speedup_by_order": dict(zip(ORDERS, speedups)),
            "k50_speedup_vs_k100": geometric_mean([
                cell["results"]["k100"]["timing"]["latency_ms_median"] /
                cell["results"]["k50"]["timing"]["latency_ms_median"] for cell in (forward, reverse)]),
            "k50_throughput_ratio_vs_official_dense": geometric_mean([
                cell["results"]["k50"]["timing"]["throughput_images_per_second_median"] /
                cell["results"]["dense"]["timing"]["throughput_images_per_second_median"] for cell in (forward, reverse)]),
        }
    drop = official["metrics"]["auc"] - sparse["metrics"]["auc"]
    speed = efficiency["8"]["k50_speedup_vs_official_dense"]
    accuracy_pass = drop < AUC_DROP_LIMIT or math.isclose(drop, AUC_DROP_LIMIT, rel_tol=0, abs_tol=1e-12)
    efficiency_pass = speed > B8_SPEEDUP_MIN or math.isclose(speed, B8_SPEEDUP_MIN, rel_tol=0, abs_tol=1e-12)
    deltas = {
        name: {key: sparse["metrics"][key] - baseline["metrics"][key] for key in ("auc", "avg_l2", "min_l2")}
        for name, baseline in (("vs_official_dense", official), ("vs_matched_dense", matched))
    }
    return {
        "task": "T4", "accuracy": {name: item["metrics"] for name, item in accuracy.items()},
        "k50_deltas": deltas, "efficiency": efficiency,
        "gate": {"accuracy_basis": "K50 vs official dense", "auc_drop_limit": AUC_DROP_LIMIT,
                 "observed_auc_drop": drop, "accuracy_pass": accuracy_pass,
                 "speedup_basis": "B8 synchronized latency; geometric mean of within-order ratios",
                 "b8_speedup_min": B8_SPEEDUP_MIN, "observed_b8_speedup": speed, "efficiency_pass": efficiency_pass,
                 "passed": accuracy_pass and efficiency_pass},
    }


def render_comparison(report):
    lines = ["# T4 · GazeFollow main experiment", "", "| Model | AUC | Avg L2 | Min L2 |", "|---|---:|---:|---:|"]
    for name, metrics in report["accuracy"].items():
        lines.append(f"| {name} | {metrics['auc']:.6f} | {metrics['avg_l2']:.6f} | {metrics['min_l2']:.6f} |")
    lines += ["", "| Batch | K50/dense latency speedup | K50/K100 speedup | K50/dense throughput ratio |",
              "|---|---:|---:|---:|"]
    for batch, values in report["efficiency"].items():
        lines.append(f"| {batch} | {values['k50_speedup_vs_official_dense']:.3f}x | {values['k50_speedup_vs_k100']:.3f}x | {values['k50_throughput_ratio_vs_official_dense']:.3f}x |")
    gate = report["gate"]
    lines += ["", f"- AUC drop vs official dense: {gate['observed_auc_drop']:.6f}; limit: {AUC_DROP_LIMIT}.",
              f"- B8 latency speedup: {gate['observed_b8_speedup']:.3f}x; minimum: {B8_SPEEDUP_MIN}x.",
              f"- Gate passed: {gate['passed']}.", ""]
    return "\n".join(lines)
