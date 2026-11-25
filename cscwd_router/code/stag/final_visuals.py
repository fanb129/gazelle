"""Export genuine predictions and routing arrays; rendering is a separate CPU step."""

import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gazelle.dataloader import load_data_gazefollow, load_data_vat
from gazelle.utils import gazefollow_l2, vat_l2
from cscwd_router.code.data_prep.final_queries import cohort
from cscwd_router.code.stag import final_io as io, final_multipeople as multi
from cscwd_router.code.stag import dinov3_efficiency as efficiency
from cscwd_router.code.stag.dinov3_run import atomic_json
from scripts.train_coverage_router import stack_predictions


def error_metrics(heatmap, head, dataset):
    if not head["inout"]:
        return None
    if dataset == "gazefollow":
        avg, minimum = gazefollow_l2(heatmap, head["gazex_norm"], head["gazey_norm"])
        return {"avg_l2": float(avg), "min_l2": float(minimum)}
    return {"l2": float(vat_l2(heatmap, head["gazex_norm"][0], head["gazey_norm"][0]))}


def scalar_error(prediction):
    metrics = prediction["errors"][0]
    return metrics.get("avg_l2", metrics.get("l2"))


def choose_error_examples(predictions, count):
    if len(predictions) < count:
        raise ValueError("candidate pool is smaller than the requested examples")
    order = sorted(range(len(predictions)), key=lambda index: (scalar_error(predictions[index]), index))
    low = (count + 1) // 2
    high = count - low
    return [(index, "lower_error") for index in order[:low]] + [(index, "higher_error") for index in order[-high:] if high]


def candidate_queries(records, seed, count):
    result = []
    for index, record in enumerate(records):
        eligible = [i for i, head in enumerate(record["heads"]) if head["inout"]]
        if eligible:
            head = min(eligible, key=lambda h: hashlib.sha256(f"{seed}\0{record['path']}\0{h}".encode()).hexdigest())
            result.append((index, head))
    result.sort(key=lambda pair: hashlib.sha256(f"{seed}\0{records[pair[0]]['path']}".encode()).hexdigest())
    return result[:count]


def person_variation(prediction):
    """Mean KL to the person mixture (Jensen-Shannon divergence), in nats."""
    probabilities = np.asarray(prediction["support_probs"], dtype=np.float64).reshape(len(prediction["support_probs"]), -1)
    probabilities = probabilities / probabilities.sum(1, keepdims=True)
    mixture = probabilities.mean(0, keepdims=True)
    return float(np.sum(probabilities * np.log((probabilities + 1e-20) / (mixture + 1e-20)), axis=1).mean())


def multi_candidates(records, selection, seed, count, minimum_inframe):
    eligible = [index for index, record in enumerate(records)
                if sum(record["heads"][h]["inout"] for h in selection["entries"][index]["head_indices"]) >= minimum_inframe]
    return sorted(eligible, key=lambda index: hashlib.sha256(
        f"{seed}\0{records[index]['path']}".encode()).hexdigest())[:count]


@torch.inference_mode()
def predict(model, transform, data_path, record, head_indices, dataset, device):
    heads = [record["heads"][index] for index in head_indices]
    with Image.open(Path(data_path) / record["path"]) as image:
        tensor = transform(image.convert("RGB"))
    output = model({"images": tensor.unsqueeze(0).to(device), "bboxes": [[head["bbox_norm"] for head in heads]]})
    maps, scores = stack_predictions(output)
    if not torch.isfinite(maps).all() or (scores is not None and not torch.isfinite(scores).all()):
        raise FloatingPointError("non-finite visualization predictions")
    maps = maps.float().cpu()
    points = []
    for heatmap in maps:
        index = int(heatmap.flatten().argmax())
        points.append([(index % heatmap.shape[1]) / heatmap.shape[1], (index // heatmap.shape[1]) / heatmap.shape[0]])
    result = {"heatmaps": np.round(maps.numpy(), 6).tolist(), "prediction_xy": points,
              "inout_scores": scores.float().cpu().tolist() if scores is not None else None,
              "errors": [error_metrics(heatmap, head, dataset) for heatmap, head in zip(maps, heads)]}
    if output.get("routing") is not None:
        routing = output["routing"]
        result.update(support_probs=np.round(routing.support_probs.float().cpu().numpy(), 8).tolist(),
            union_probs=np.round(routing.image_union_probs.float().cpu().numpy(), 8).tolist(),
            kept_masks=routing.image_hard_masks.cpu().int().tolist(),
            keep_indices=routing.image_keep_indices.cpu().tolist(), person_to_image=routing.person_to_image.cpu().tolist())
    return result


def case_data(root, figure_root, data_path, record, head_indices, case_id, dataset, kind, predictions, **metadata):
    image_path = figure_root / "assets" / f"{case_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(Path(data_path) / record["path"]) as image:
        image = image.convert("RGB")
        original_size = list(image.size)
        image.thumbnail((768, 768))
        image.save(image_path, quality=92)
    heads = [record["heads"][index] for index in head_indices]
    case = {"id": case_id, "kind": kind, "dataset": dataset, "sample_path": record["path"],
        "original_image_size": original_size, "head_indices": head_indices,
        "image_relative_path": os.path.relpath(image_path, root), "source_image": io.reference(Path(data_path) / record["path"]),
        "heads": [{key: head[key] for key in ("bbox_norm", "gazex_norm", "gazey_norm", "inout")} for head in heads],
        "predictions": predictions, **metadata}
    path = root / "cases" / f"{case_id}.json"
    atomic_json(path, case)
    return {"id": case_id, "path": str(path.relative_to(root)), "sha256": io.reference(path)["sha256"]}, {
        f"case_{case_id}": io.reference(path), f"image_{case_id}": io.reference(image_path)}


def prepare(args):
    if args.seed != 0 or args.examples_per_dataset < 2 or args.candidate_count < args.examples_per_dataset:
        raise ValueError("use seed0 and at least two examples within the candidate pool")
    sources = {}
    for key, path in (("gf", args.gf_summary), ("vat", args.vat_summary), ("multi", args.multi_summary)):
        sources[key] = io.verified_source(path, args.allow_closed_gf)
    if sources["gf"][0]["dataset"] != "gazefollow" or any(sources[k][0]["dataset"] != "vat" for k in ("vat", "multi")):
        raise ValueError("visualization sources must be GF, VAT and VAT multi-query models")
    records = {"gf": load_data_gazefollow(args.gf_data_path / "test_preprocessed.json"),
               "vat": load_data_vat(args.vat_data_path / "test_preprocessed.json", sample_rate=1)}
    multi_records, selection = cohort(args.vat_data_path, args.seed)
    records["multi"] = multi_records
    candidates = {key: candidate_queries(records[key], args.seed, args.candidate_count) for key in ("gf", "vat")}
    if not 1 <= args.multi_examples <= args.multi_candidate_count:
        raise ValueError("multi examples must fit within the multi candidate pool")
    selected_multi = multi_candidates(multi_records, selection, args.seed, args.multi_candidate_count, args.multi_min_inframe)
    if len(selected_multi) < args.multi_examples:
        raise ValueError("not enough multi-query frames satisfy the requested in-frame count")
    rgb_refs = {}
    for key in ("gf", "vat"):
        data_path = args.gf_data_path if key == "gf" else args.vat_data_path
        for index, _ in candidates[key]:
            path = data_path / records[key][index]["path"]
            rgb_refs[str(path.resolve())] = io.reference(path)["sha256"]
    for index in selected_multi:
        path = args.vat_data_path / multi_records[index]["path"]
        rgb_refs[str(path.resolve())] = io.reference(path)["sha256"]
    for key, data_path in (("gf", args.gf_data_path), ("vat", args.vat_data_path), ("multi", args.vat_data_path)):
        if io.reference(data_path / "test_preprocessed.json") != sources[key][1]["data"]["test_annotation"]:
            raise ValueError("visualization annotation differs from the selected source")
    request = {"protocol": "genuine_routing_visualization_exports_v1", "seed": args.seed,
        "source_summaries": {key: io.reference(path) for key, path in (("gf", args.gf_summary), ("vat", args.vat_summary), ("multi", args.multi_summary))},
        "checkpoints": {key: value[0]["artifacts"]["best_test_checkpoint"] for key, value in sources.items()},
        "backbones": {key: value[0]["artifacts"]["backbone"] for key, value in sources.items()},
        "official_checkpoints": {key: io.reference(args.official_checkpoint_dir /
            f"gazelle_dinov2_vit{sources[key][0]['size']}14{'_inout' if key == 'vat' else ''}.pt") for key in ("gf", "vat")},
        "official_backbones": {key: io.reference(io.ROOT / "checkpoints" / f"dinov2_vit{sources[key][0]['size']}14_pretrain.pth") for key in ("gf", "vat")},
        "annotation_sha256": {key: value[1]["data"]["test_annotation"] for key, value in sources.items()},
        "candidate_count": args.candidate_count, "examples_per_dataset": args.examples_per_dataset,
        "multi_examples": args.multi_examples, "selection_policy": "lower_and_higher_error_within_seeded_candidate_pool",
        "multi_candidate_count": args.multi_candidate_count,
        "multi_min_inframe": args.multi_min_inframe,
        "multi_selection_policy": "highest_person_support_Jensen_Shannon_divergence_within_seeded_candidate_pool",
        "RGB_sha256": rgb_refs, "multi_selection_sha256": selection["selection_sha256"],
        "sources": {**efficiency.source_hashes(), **io.sources("cscwd_router/code/stag/final_visuals.py",
            "cscwd_router/code/scripts/export_final_visualizations.py", "cscwd_router/code/data_prep/final_queries.py",
            "cscwd_router/code/stag/final_multipeople.py")}}
    return sources, records, candidates, selected_multi, selection, request


def export(args):
    sources, records, candidates, selected_multi, selection, request = prepare(args)
    output = args.report_root / "manifest.json"
    if args.check_only:
        result = {"checked": True, "single_candidates": {key: len(value) for key, value in candidates.items()},
                  "multi_candidates": len(selected_multi), "model_forward_executed": False}
        print(json.dumps(result), flush=True)
        return result
    old = io.existing(output, request, args.resume)
    if old:
        return old
    cases, artifacts = [], {}
    for key, dataset, data_path in (("gf", "gazefollow", args.gf_data_path), ("vat", "vat", args.vat_data_path)):
        source, _, payload = sources[key]
        model, transform = io.build_router(source, payload)
        model.to(args.device)
        pool = [predict(model, transform, data_path, records[key][index], [head], dataset, args.device) for index, head in candidates[key]]
        chosen = choose_error_examples(pool, args.examples_per_dataset)
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        official_path = args.official_checkpoint_dir / f"gazelle_dinov2_vit{source['size']}14{'_inout' if key == 'vat' else ''}.pt"
        official, transform = efficiency.build_variant(source, payload, official_path, "official")
        official.to(args.device)
        for order, (candidate, bucket) in enumerate(chosen):
            index, head = candidates[key][candidate]
            baseline = predict(official, transform, data_path, records[key][index], [head], dataset, args.device)
            ref, refs = case_data(args.report_root, args.figure_root, data_path, records[key][index], [head],
                f"{key}_single_{order:02d}", dataset, "single", {"official": baseline, "union": pool[candidate]},
                error_bucket=bucket, pool_size=len(pool), source_selection=source["selection"])
            cases.append(ref)
            artifacts.update(refs)
        del official
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    source, _, payload = sources["multi"]
    predictions = {index: {} for index in selected_multi}
    variation = {}
    for variant in ("union", "independent"):
        model, transform = multi.build_variant(source, payload, None, variant)
        model.to(args.device)
        for index in selected_multi:
            heads = selection["entries"][index]["head_indices"]
            predictions[index][variant] = predict(model, transform, args.vat_data_path, records["multi"][index], heads, "vat", args.device)
        if variant == "union":
            variation = {index: person_variation(predictions[index]["union"]) for index in selected_multi}
            selected_multi = sorted(selected_multi, key=lambda index: (-variation[index], index))[:args.multi_examples]
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    for order, index in enumerate(selected_multi):
        ref, refs = case_data(args.report_root, args.figure_root, args.vat_data_path, records["multi"][index],
            selection["entries"][index]["head_indices"], f"vat_multi_{order:02d}", "vat", "multi", predictions[index],
            source_selection=source["selection"], person_support_JS_nats=variation[index],
            multi_selection_policy=request["multi_selection_policy"], multi_candidate_count=len(predictions),
            multi_min_inframe=request["multi_min_inframe"])
        cases.append(ref)
        artifacts.update(refs)
    result = {"request": request, "cases": cases, "precision": "fp32", "array_rounding": "heatmaps6/support8 decimals; prediction coordinates and errors retained before rounding",
              "rendering": "CPU-only renderer; source/selection unchanged", "runtime": io.runtime(args.device), "artifacts": artifacts}
    result = io.finish(output, result, args.provenance, "Final genuine routing visualization exports")
    print(f"COMPLETE {output}", flush=True)
    return result
