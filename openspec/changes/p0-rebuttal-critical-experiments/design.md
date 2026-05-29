## Context

The rebuttal needs evidence for four repeated reviewer concerns: missing GOO-real OOD evaluation, no head-box noise robustness test, incomplete efficiency reporting, and a misleading parameter table that currently emphasizes trainable/head parameters without clearly separating total frozen-VFM parameters. The local workspace is for code and proposal artifacts only; the user will run real experiments manually on `fb@3090.lab`, where GOO-real is already downloaded at `/newhome/fb/dataset/gooreal_data`.

Current code has GazeFollow/VAT evaluation scripts, VAT/GazeFollow preprocessors, Crowd subset builders, and `scripts/compute_flops.py`. P0 should add honest, metrics-first experiment scripts and avoid using visualization-only outputs or any hand-edited results.

## Goals / Non-Goals

**Goals:**

- Prepare GOO-real preprocessing/evaluation scripts that adapt GOO-real annotations into the existing normalized `path`, `heads`, `bbox_norm`, `gazex_norm`, `gazey_norm`, and `inout` schema.
- Prepare deterministic bbox-noise robustness evaluation for baseline vs GazeSpot at `0%`, `5%`, `10%`, and `20%` translation/scale jitter.
- Extend complexity measurement from `scripts/compute_flops.py` to report trainable params, total params, MACs/FLOPs, latency ms/img, and FPS in JSON/CSV/Markdown-friendly forms.
- Provide server commands and rebuttal table templates with placeholders only.
- Make every output label distinguish `Trainable Params` from `Total Params`, including frozen VFM parameters.

**Non-Goals:**

- Do not train new models unless the user later requests it.
- Do not implement P1 ablations such as Gaussian masks, CoordConv, FPN fusion, over-smoothing feature analysis, or multi-seed significance studies.
- Do not implement P2 rebuttal writing beyond P0 result table templates and command documentation.
- Do not evaluate ChildPlay in this change because the README says ChildPlay-gaze is not downloaded.
- Do not invent, estimate, or pre-fill experimental numbers.

## Decisions

### Decision: Use the Existing Normalized Annotation Schema

GOO-real preprocessing should output JSON shaped like the current GazeFollow evaluator expects: each image item has `path`, `width`, `height`, and `heads`, and each head has normalized bbox/gaze fields plus `inout`. This allows GOO-real to reuse the same model call pattern as `scripts/eval_gazefollow.py` and avoids adding a parallel dataset abstraction during rebuttal crunch time.

Alternative considered: evaluate directly from GOO-real pickle files. That is faster to write for one script, but it would make robustness evaluation and table serialization less reusable.

### Decision: Separate Metrics-Only Evaluation from Visualization

The P0 scripts should emit structured result files under an explicit output directory, for example `rebuttal/results/p0/*.json`, `*.csv`, and `*.md`. Visualizations are optional and must not be used as the source of quantitative tables. Existing visualization helpers in `scripts/eval_gazefollow.py` and `scripts/eval_vat.py` can be referenced, but new P0 scripts should not include GT-blended or beautified heatmaps in metric paths.

Alternative considered: modify the existing eval scripts in place. That risks mixing rebuttal metrics with current visual comparison behavior, so the safer path is new metrics scripts plus small shared helpers.

### Decision: Deterministic Bbox Jitter

Head-box perturbation should use deterministic seeded jitter and report the seed. For each bbox `[xmin, ymin, xmax, ymax]`, sample translation offsets as a fraction of box width/height and scale changes as a fraction of box size, clip to `[0, 1]`, and preserve valid ordering with a minimum box size guard. The `0%` level must be the original boxes and should reproduce the non-noise metric path.

Alternative considered: pre-generate perturbed JSON files. Runtime perturbation keeps disk output smaller and makes it easier to run all levels from one command, while an optional `--save_perturbed_annotations` can be added if inspection is needed.

### Decision: Report Both Trainable and Total Parameters Everywhere

Parameter reporting should use `sum(p.numel() for p in model.parameters())` for total params and `sum(p.numel() for p in model.parameters() if p.requires_grad)` for trainable params. If existing freeze flags make `requires_grad` unreliable, the implementation must explicitly freeze the model as inference code does before counting and can additionally report `non_backbone_params` as a diagnostic, not as the table's only parameter count.

Alternative considered: keep the current `Head Params` label. Reviewers specifically called this misleading, so the output label must be changed to `Trainable Params` and accompanied by `Total Params`.

### Decision: Measure RTX 3090 Latency with Warmup, Synchronization, and Batch-1 Defaults

The complexity script should benchmark on CUDA with `torch.cuda.synchronize()`, warmup iterations, measured iterations, and batch size `1` by default for ms/img and FPS. It should accept batch-size overrides but the rebuttal table should report the batch-1 default unless the user explicitly chooses another setting. The output must include device name so the table can state `RTX 3090`.

Alternative considered: rely only on THOP MACs. Reviewers asked for deployment metrics, so latency/FPS must be measured on the server GPU.

## Expected Changed Files

- `data_prep/preprocess_gooreal.py`: convert server GOO-real pickle/archive data into the normalized evaluation JSON schema.
- `scripts/eval_gooreal.py`: evaluate baseline and GazeSpot on the preprocessed GOO-real split(s).
- `scripts/eval_bbox_noise.py`: evaluate baseline and GazeSpot under deterministic bbox perturbations on VAT/Crowd and optionally GazeFollow.
- `scripts/compute_flops.py`: extend to structured complexity and latency/FPS reporting.
- `gazelle/utils.py` or `gazelle/eval_utils.py`: add reusable bbox jitter, metric aggregation, parameter counting, and result serialization helpers.
- `rebuttal/p0_experiment_commands.md`: document server commands and table templates.

## Server Run Commands

All commands are intended for `fb@3090.lab` after the user pulls the branch. Paths are concrete where known and placeholders are explicit where checkpoint locations may differ. The commands must write real outputs; the rebuttal tables must remain blank until those outputs exist.

```bash
ssh fb@3090.lab
cd /home/fb/src/paper/gazelleV1
git checkout v1
git pull
```

GOO-real preprocessing and evaluation:

```bash
/home/fb/anaconda3/envs/py310/bin/python data_prep/preprocess_gooreal.py \
  --data_path /newhome/fb/dataset/gooreal_data \
  --output_json /newhome/fb/dataset/gooreal_data/gooreal_test_preprocessed.json

/home/fb/anaconda3/envs/py310/bin/python scripts/eval_gooreal.py \
  --data_path /newhome/fb/dataset/gooreal_data \
  --json_path /newhome/fb/dataset/gooreal_data/gooreal_test_preprocessed.json \
  --base_ckpt /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitb_v0/2026-03-19_16-40-15/epoch_14.pt \
  --spot_ckpt /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt \
  --batch_size 64 \
  --output rebuttal/results/p0/gooreal_eval.json
```

VAT/Crowd bbox-noise robustness:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/eval_bbox_noise.py \
  --dataset vat \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --json_path /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_gt4.json \
  --base_ckpt /home/fb/src/paper/gazelleV1/experiments/train_vat_vitb_v0/2026-03-20_22-30-50/epoch_7.pt \
  --spot_ckpt /home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt \
  --jitter_levels 0 5 10 20 \
  --seed 3106 \
  --batch_size 16 \
  --output rebuttal/results/p0/vat_crowd_bbox_noise.json
```

Optional GazeFollow bbox-noise robustness:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/eval_bbox_noise.py \
  --dataset gazefollow \
  --data_path /newhome/fb/dataset/gazefollow_extended \
  --json_path /newhome/fb/dataset/gazefollow_extended/test_preprocessed.json \
  --base_ckpt /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitb_v0/2026-03-19_16-40-15/epoch_14.pt \
  --spot_ckpt /home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt \
  --jitter_levels 0 5 10 20 \
  --seed 3106 \
  --batch_size 64 \
  --output rebuttal/results/p0/gazefollow_bbox_noise.json
```

Complexity/latency/FPS:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/compute_flops.py \
  --device cuda \
  --warmup_iters 50 \
  --measure_iters 200 \
  --batch_size 1 \
  --output rebuttal/results/p0/complexity_latency.json
```

## Rebuttal Table Templates

The following tables are templates only. Numeric cells must be filled only from server outputs.

GOO-real OOD evaluation:

| Dataset | Method | AUC ↑ | Avg L2 ↓ | Min L2 ↓ | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| GOO-real | Baseline (DINOv3 last-layer) | TBD | TBD | TBD | filled from server output |
| GOO-real | GazeSpot | TBD | TBD | TBD | filled from server output |

Head-box noise robustness:

| Dataset/Split | Jitter | Method | AUC ↑ | L2 ↓ | In/Out AP ↑ |
| --- | ---: | --- | ---: | ---: | ---: |
| VAT Crowd | 0% | Baseline | TBD | TBD | TBD |
| VAT Crowd | 0% | GazeSpot | TBD | TBD | TBD |
| VAT Crowd | 5% | Baseline | TBD | TBD | TBD |
| VAT Crowd | 5% | GazeSpot | TBD | TBD | TBD |
| VAT Crowd | 10% | Baseline | TBD | TBD | TBD |
| VAT Crowd | 10% | GazeSpot | TBD | TBD | TBD |
| VAT Crowd | 20% | Baseline | TBD | TBD | TBD |
| VAT Crowd | 20% | GazeSpot | TBD | TBD | TBD |

Complexity and runtime:

| Method | Input | Trainable Params ↓ | Total Params ↓ | MACs/FLOPs ↓ | Latency ms/img ↓ | FPS ↑ | Device |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline (GF) | 448 | TBD | TBD | TBD | TBD | TBD | RTX 3090 |
| GazeSpot (GF) | 512 | TBD | TBD | TBD | TBD | TBD | RTX 3090 |
| Baseline (VAT) | 448 | TBD | TBD | TBD | TBD | TBD | RTX 3090 |
| GazeSpot (VAT) | 512 | TBD | TBD | TBD | TBD | TBD | RTX 3090 |

## Risks / Trade-offs

- [GOO-real annotation shape differs from assumptions] → Preprocessing must inspect pickle keys and fail with a clear message listing discovered keys rather than silently producing wrong labels.
- [Existing checkpoint paths differ on the server] → Commands use current repo defaults where available and keep checkpoint args explicit so the user can substitute paths.
- [THOP cannot count all ViT operations exactly] → Report MACs/FLOPs as tool-measured estimates, include the tool name in output metadata, and rely on latency/FPS for deployment evidence.
- [Bbox jitter randomness changes results] → Require fixed seed and include seed/jitter definition in output metadata.
- [Runtime too high during rebuttal window] → VAT Crowd is required for robustness; GazeFollow noise is optional. Batch-1 latency is required; extra batch sizes are optional.
