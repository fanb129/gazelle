"""Verify four GOO test cells against raw predictions, then write the paper table."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    def artifact_path(item):
        return (args.artifact_root_mirror / item["path"].lstrip("/")
                if args.artifact_root_mirror else Path(item["path"]))

    reports = {}
    references = []
    identities = None
    annotation = None
    for size in ("b", "l"):
        for variant in ("k50", "official"):
            path = args.report_root / f"vit{size}" / variant / "summary.json"
            report = json.loads(path.read_text())
            request = report["request"]
            if not report["completed"] or (request["size"], request["variant"]) != (size, variant):
                raise ValueError("GOO matrix contains an incomplete or misidentified cell")
            if request["target_training"] or request["target_checkpoint_selection"]:
                raise ValueError("GOO result is not fixed-checkpoint zero-shot inference")
            for artifact in report["artifacts"].values():
                if sha256(artifact_path(artifact)) != artifact["sha256"]:
                    raise ValueError("GOO prediction artifact SHA256 mismatch")
            with gzip.open(artifact_path(report["artifacts"]["per_query_predictions"]), "rt") as handle:
                rows = [json.loads(line) for line in handle]
            keys = [(r["index"], r["path"], r["bbox_norm"], r["gaze"]) for r in rows]
            if identities is None:
                identities = keys
                annotation = request["annotation"]
            if keys != identities or request["annotation"] != annotation:
                raise ValueError("GOO cells do not share identical test queries and GT boxes")
            if len(rows) != 2146 or [r["index"] for r in rows] != list(range(2146)):
                raise ValueError("GOO result does not cover the official full test split")
            for metric in ("auc", "l2"):
                actual = math.fsum(r[metric] for r in rows) / len(rows)
                if not math.isclose(actual, report["metrics"][metric], rel_tol=0, abs_tol=1e-12):
                    raise ValueError("GOO summary differs from per-query metrics")
            if variant == "k50":
                coverage = sum(r["gt_point_kept"] for r in rows)/len(rows)
                if coverage != report["metrics"]["routing_gt_point_coverage"]:
                    raise ValueError("GOO coverage differs from raw predictions")
            reports[f"vit{size}/{variant}"] = report["metrics"]
            references.append({"path": str(path.resolve()), "sha256": sha256(path)})
    delta = {f"vit{size}": {k: reports[f"vit{size}/k50"][k] - reports[f"vit{size}/official"][k]
                              for k in ("auc", "l2")} for size in ("b", "l")}
    summary = {"completed": True, "protocol": "GOO_Real_full_test_zero_shot_v1", "reports": references,
               "metrics": reports, "router_minus_official": delta,
               "verification": "4 artifact pairs hashed; identical 2146 query/GT records; raw means matched within 1e-12"}
    (args.report_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# GOO-Real fixed-checkpoint zero-shot evaluation", "",
             "Full 2,146-image official test split; GF-trained weights; real GT head boxes; FP32; seed 0.", "",
             "| Model | AUC ↑ | L2 ↓ | GT point coverage ↑ |", "|---|---|---|---|"]
    for size in ("b", "l"):
        for variant in ("official", "k50"):
            row = reports[f"vit{size}/{variant}"]
            name = f"Official GazeLLE-{size.upper()} (re-evaluated)" if variant == "official" else f"Router-{size.upper()} K50"
            coverage = "—" if variant == "official" else f"{row['routing_gt_point_coverage']:.6f}"
            lines.append(f"| {name} | {row['auc']:.6f} | {row['l2']:.6f} | {coverage} |")
    lines += ["", summary["verification"], "",
              "Published GazeLLE numbers use its original evaluation preparation; distinguish them from these matched-box re-evaluations."]
    (args.report_root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-root", type=Path, required=True)
    p.add_argument("--artifact-root-mirror", type=Path,
                   help="local mirror of server /; allows offline raw-prediction verification")
    run(p.parse_args())
