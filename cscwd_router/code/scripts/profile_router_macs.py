"""Count K100/K50 Conv, Linear and attention MACs on the native graph, using CPU."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from cscwd_router.code.stag import complexity


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--native-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def run(args):
    torch.set_num_threads(args.threads)
    native_hash = complexity.native_source_hash(args.native_repo)
    cells = []
    for path in args.summaries:
        source = json.loads(path.read_text())
        manifest = json.loads(path.with_name("manifest.json").read_text())
        if native_hash != manifest["sources"]["native_dinov3_source_tree"]:
            raise ValueError("native source differs from the training manifest")
        graph_files = ("gazelle/backbone.py", "gazelle/model.py", "gazelle/routing/backbone.py",
                       "gazelle/routing/model.py", "gazelle/routing/router.py", "gazelle/utils.py",
                       "gazelle/ablation_variants.py")
        graph_sources = {}
        for name in graph_files:
            expected = manifest["sources"].get(name)
            if expected is None:
                original = subprocess.check_output(["git", "show", f"{source['runtime']['git_commit']}:{name}"], cwd=ROOT)
                expected = hashlib.sha256(original).hexdigest()
            if complexity.sha256(ROOT / name) != expected:
                raise ValueError(f"profile source differs from training: {name}")
            graph_sources[name] = expected
        torch.manual_seed(args.seed)
        started = time.monotonic()
        model = complexity.build_architecture(source, args.native_repo)
        inputs = {"images": torch.randn(1, 3, 512, 512), "bboxes": [[[.1, .1, .3, .3]]]}
        results = {}
        for variant, keep in (("k100", 1.), ("k50", .5)):
            model.router.keep_ratio = keep
            results[variant] = complexity.profile(model, inputs)
        cell = {"dataset": source["dataset"], "size": source["size"],
            "source_summary": {"path": str(path.resolve()), "sha256": complexity.sha256(path)},
            "selected_checkpoint_reference": source["artifacts"]["best_test_checkpoint"],
            "model_config": source["model_config"], "profiled_graph_sources": graph_sources,
            "source_completed": source["completed"],
            "source_closed_by_user": source.get("closed_by_user", False),
            "architecture_only_seconds": time.monotonic() - started, "results": results}
        cells.append(cell)
        print(f"COMPLETE {source['dataset']} vit{source['size']} CPU MACs", flush=True)
        del model, inputs
        gc.collect()
    report = {"completed": True, "protocol": "native_router_cpu_macs_v1",
        "runtime": {"device": "cpu", "python": platform.python_version(), "torch": str(torch.__version__),
            "threads": args.threads, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "convention": {"mac_flops": 2, "scope": "complete model forward",
            "counted": ["Conv2d", "ConvTranspose2d", "Linear", "attention QK^T", "attention AV"],
            "not_counted": ["bias additions", "normalization", "activations", "softmax", "RoPE", "pooling",
                            "Top-K/sort", "gather/scatter", "memory copies", "elementwise geometry/masks"],
            "weights": "untrained native architecture; operation shapes depend on fixed K, not weight values",
            "inputs": "one synthetic RGB tensor and one synthetic head box; no accuracy or latency measurement",
            "training": False, "checkpoint_weights_loaded": False, "seed": args.seed},
        "sources": {"native_dinov3_source_tree": native_hash, **{name: complexity.sha256(ROOT / name) for name in (
            "cscwd_router/code/stag/complexity.py", "cscwd_router/code/scripts/profile_router_macs.py")}},
        "cells": cells, "summary": complexity.summarize(cells)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Native Router CPU MACs", "", "One image, one head query, 512x512. Conventional Conv/Linear/attention MACs; FLOPs = 2 MACs.",
        "Untrained native weights; no quality or runtime result. Both paths retain the Router and the same multilayer decoder.", "",
        "| Dataset | Size | K100 GMACs | K50 GMACs | MAC reduction |", "|---|---|---|---|---|"]
    for row in report["summary"]:
        lines.append(f"| {row['dataset']} | ViT-{row['size'].upper()} | {row['k100_gmacs']:.6f} | {row['k50_gmacs']:.6f} | {100 * row['mac_reduction_fraction']:.2f}% |")
    lines += ["", "Elementwise arithmetic, normalization, RoPE, softmax, sorting and memory movement are excluded from this MAC convention; their runtime remains included in the GPU forward benchmarks."]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    run(parse_args())
