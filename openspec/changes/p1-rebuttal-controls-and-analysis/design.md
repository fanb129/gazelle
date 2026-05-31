## Context

The current paper argues that GazeSpot improves crowded gaze target estimation through two mechanisms: Geometry-Guided Spatial Focus (GGSF), a learned head-conditioned multiplicative mask, and Scale-Aware Semantic Aggregation (SASA), a learned dynamic multi-layer fusion module. The submitted ablation already covers multi-layer raw concatenation, SASA, and SASA+GGSF, but Reviewer 7pTL explicitly asks for stronger alternatives: Gaussian masks, CoordConv-style coordinate channels, hand-designed cone masks, FPN-style fusion, selected layers, quantitative over-smoothing evidence, and repeated-run reliability. Reviewer CoU9 also questions whether GGSF learns a meaningful spotlight and whether SASA is more than classic multi-scale fusion.

Local code changes are prepared in this workspace only. The user will pull the branch and run real training/evaluation on `fb@3090.lab`. No experimental number may be invented locally. P0 has already handled GOO-real, bbox-noise robustness, latency/FPS, and parameter reporting; P1 should only depend on those interfaces when needed and should not re-own them.

## Goals / Non-Goals

**Goals:**

- Execute P1 as a staged, server-gated workflow: complete one task section locally, run local checks, have the user sync to `fb@3090.lab`, wait for server experiment feedback, and only then continue to the next section.
- Provide controlled GGSF alternatives: no gate, fixed Gaussian head-centered mask, CoordConv-style coordinate channels, learned GGSF, and fixed cone/sector mask if feasible.
- Provide controlled SASA alternatives: raw concatenation, equal-weight fusion, FPN-style fusion, selected-layer variants, and learned SASA.
- Quantify feature over-smoothing for DINOv2 vs DINOv3 and shallow/mid/deep layers on crowded examples using required lightweight metrics and optional heavier probes.
- Provide 3-seed mean/std support for the key Crowd dense subset comparison when server budget permits.
- Emit structured JSON/CSV/Markdown outputs and rebuttal-ready table/plot templates under `rebuttal/results/p1/`.
- Provide exact post-implementation server commands that the user can run on `fb@3090.lab`.

**Non-Goals:**

- Do not implement P0 GOO-real, bbox-noise robustness, latency/FPS, or parameter-reporting work again.
- Do not evaluate ChildPlay.
- Do not write final rebuttal prose; P2 owns response assembly and wording.
- Do not change dataset contents or claim new results before server outputs exist.
- Do not convert production GazeSpot behavior by default; P1 variants must be opt-in through CLI/config.
- Do not implement all P1 sections in one uninterrupted pass before server validation.

## Decisions

### Decision: Implement Ablation Variants as Explicit Modes

Add explicit mode arguments instead of many boolean combinations:

- `--spatial_prior {none,ggsf,fixed_gaussian,coordconv,fixed_sector}`
- `--fusion {raw_concat,equal_weight,sasa,fpn,selected_layers}`
- `--selected_layers 2,5,8,11` or named presets such as `shallow`, `mid`, `deep`, `last`

The model should keep existing `--use_sasa` and `--use_ggsf` compatibility, but the rebuttal runner should use the mode arguments and map them to the current code path. This makes result metadata unambiguous.

Alternative considered: add separate scripts for every variant. That would reduce branch complexity per script but make result aggregation and seed sweeps brittle.

### Decision: Gate Every Section on Server Feedback

The apply phase should treat each numbered task section as its own mini-delivery. The developer or subagent completes one section, runs local verification for that section, reports changed files and exact server commands, then stops. The user syncs code to `fb@3090.lab`, runs the section-specific experiment or smoke command, and returns logs/results. The next section begins only after the user confirms that the server result is acceptable or provides a concrete fix request.

This staged workflow is required because the local environment is only for editing code, while true training/evaluation happens on the server. It also prevents hidden coupling between controls: if spatial-prior variants fail on server, fusion controls should not be layered on top before the failure is understood.

Alternative considered: implement all scripts first and run one final server pass. That is faster on paper, but it delays discovering server-only failures and does not match the user's intended subagent workflow.

### Decision: Keep Fixed Spatial Priors Deterministic

Fixed spatial priors should be generated from normalized head boxes on the feature grid:

- `none`: all-ones gate.
- `fixed_gaussian`: isotropic or mildly anisotropic Gaussian centered at the head center, with sigma derived from head box size and documented in metadata.
- `coordconv`: append normalized coordinate and relative head-coordinate channels before fusion/linear projection; if full append is too invasive, implement as a light projection that uses `[x, y, dx, dy, w, h]` channels.
- `fixed_sector`: deterministic wedge/sector mask anchored at head center. Because this repository does not estimate head pose or gaze vector, the feasible fallback is a symmetric downward/outward or image-center-biased sector template with its limitation recorded. If this risks becoming arbitrary, mark it optional and report that the stronger non-learned controls are Gaussian and CoordConv.

Alternative considered: learn Gaussian sigma or sector direction. That would blur the comparison with GGSF and weaken the “simple fixed prior” control.

### Decision: Compare Fusion Under the Same Backbone, Decoder, and Training Budget

Fusion variants should share DINO backbone, decoder, input resolution, optimizer, epoch count, and dataset split. Only the fusion block changes:

- `raw_concat`: existing multi-layer concatenate + 1x1 projection.
- `equal_weight`: multiply each layer by `1/K`, concatenate, then project.
- `sasa`: existing learned scale-attention weights.
- `fpn`: top-down lateral 1x1 projections with interpolation/addition, then final projection.
- `selected_layers`: use a subset such as shallow-only, mid-only, deep-only/last-only, shallow+mid, mid+deep; keep selected subsets smaller if runtime is tight.

The required rebuttal table should prioritize VAT Crowd dense because the contested claim is crowded-scene robustness. GazeFollow is optional unless a quick run is already available.

Alternative considered: evaluate only checkpoints without retraining. This is acceptable for feature analysis but not for architectural ablations because control variants need fair training from the same initialization.

### Decision: Required Feature Metrics Are Lightweight and Label-Minimal

Required over-smoothing metrics:

- `crowd_token_cosine`: mean pairwise cosine similarity among feature tokens in crowded person/head regions; higher values indicate more homogeneous features.
- `inter_person_boundary_separability`: cosine distance or contrast between tokens inside target/neighbor head boxes and adjacent boundary bands when multiple annotated heads are available.
- `foreground_background_contrast`: cosine or L2 distance between person/head tokens and sampled background tokens.
- `layer_summary`: report the metrics for DINOv2 and DINOv3 at shallow/mid/deep/last layers.

Optional metrics if runtime and annotation coverage allow:

- `layerwise_probe`: train a small frozen-feature probe for in-head/person/background or gaze heatmap proxy prediction.
- `attention_entropy_or_rank`: compute token covariance effective rank or entropy as an additional smoothness indicator.
- `full VAT/GazeFollow sweep`: run analysis beyond Crowd dense examples.

Alternative considered: use PCA visualizations only. The reviewers directly said PCA is insufficient, so PCA may be kept only as a plot supplement, not the quantitative claim.

### Decision: Statistical Reliability Is a Commanded, Table-Ready Seed Sweep

Add a seed-sweep runner that can train/evaluate the key variants with seeds `3106`, `3107`, and `3108`, then summarize mean/std. Required support is command and aggregation support; running all three seeds is required only for the key Crowd dense comparison if server time permits.

Required 3-seed target:

- Baseline DINOv3 last-layer or raw-concat baseline vs full GazeSpot on VAT Crowd dense (`>=4` or `>4`, matching available P0/Crowd JSON).

Optional 3-seed extensions:

- Best fixed Gaussian vs GGSF.
- Best FPN/equal-weight variant vs SASA.
- GazeFollow or full VAT.

Alternative considered: bootstrap confidence intervals from one checkpoint. Bootstrapping is useful as optional support, but it does not address training variance as directly as repeated seeds.

## Expected Changed Files

- `gazelle/model.py`: add spatial/fusion mode plumbing for DINOv3 GazeSpot variants while preserving default behavior.
- `gazelle/model_dinov2.py`: mirror mode plumbing needed for DINOv2 feature analysis or selected DINOv2 controls.
- `gazelle/ablation_variants.py`: new helper module for fixed spatial masks, CoordConv feature adapters, FPN/equal-weight fusion, selected-layer parsing, and variant metadata.
- `scripts/train_gazefollow.py`: accept/rewrite variant CLI args and record metadata.
- `scripts/train_vat.py`: accept/rewrite variant CLI args, seed, and metadata.
- `scripts/eval_gazefollow.py`: support evaluating a named variant checkpoint without visualization-only paths contaminating metrics.
- `scripts/eval_vat.py`: support evaluating a named variant checkpoint on Crowd dense JSON paths and writing structured metrics.
- `scripts/run_rebuttal_ablation.py`: orchestrate train/eval commands for selected variant groups and emit JSONL manifest.
- `scripts/analyze_feature_oversmoothing.py`: compute DINOv2/DINOv3 layer-wise feature smoothness/separability metrics.
- `scripts/summarize_p1_results.py`: convert JSON/CSV outputs into Markdown tables and plot-ready CSVs.
- `rebuttal/p1_controls_and_analysis.md`: server commands, required/optional analysis matrix, and rebuttal result templates.
- `rebuttal/results/p1/`: generated output directory for JSON/CSV/Markdown result files.

## Server Run Commands

All commands are intended for `fb@3090.lab` after implementation and after the user pulls the branch. Checkpoint/output paths are explicit; replace only checkpoint directories if the server uses different experiment names.

```bash
ssh fb@3090.lab
cd /home/fb/src/paper/gazelleV1
git checkout v1
git pull
mkdir -p rebuttal/results/p1
```

Section 1 server smoke check after variant configuration lands:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/run_rebuttal_ablation.py \
  --print_plan_only \
  --group spatial_prior \
  --dataset vat \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --crowd_json /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_ge4.json \
  --variants none fixed_gaussian coordconv ggsf \
  --seed 3106 \
  --output_dir rebuttal/results/p1/smoke_variant_config
```

Required spatial-prior ablations on VAT Crowd dense:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/run_rebuttal_ablation.py \
  --group spatial_prior \
  --dataset vat \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --crowd_json /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_ge4.json \
  --init_ckpt ./checkpoints/gazelle_dinov3_vitb16.pt \
  --variants none fixed_gaussian coordconv ggsf \
  --seed 3106 \
  --max_epochs 8 \
  --batch_size 60 \
  --output_dir rebuttal/results/p1/spatial_prior
```

Optional fixed sector spatial-prior ablation:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/run_rebuttal_ablation.py \
  --group spatial_prior \
  --dataset vat \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --crowd_json /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_ge4.json \
  --init_ckpt ./checkpoints/gazelle_dinov3_vitb16.pt \
  --variants fixed_sector \
  --seed 3106 \
  --max_epochs 8 \
  --batch_size 60 \
  --output_dir rebuttal/results/p1/spatial_prior_optional
```

Required fusion ablations on VAT Crowd dense:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/run_rebuttal_ablation.py \
  --group fusion \
  --dataset vat \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --crowd_json /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_ge4.json \
  --init_ckpt ./checkpoints/gazelle_dinov3_vitb16.pt \
  --variants raw_concat equal_weight fpn sasa \
  --seed 3106 \
  --max_epochs 8 \
  --batch_size 60 \
  --output_dir rebuttal/results/p1/fusion
```

Optional selected-layer fusion ablations:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/run_rebuttal_ablation.py \
  --group fusion \
  --dataset vat \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --crowd_json /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_ge4.json \
  --init_ckpt ./checkpoints/gazelle_dinov3_vitb16.pt \
  --variants selected_layers \
  --selected_layer_sets shallow mid deep shallow_mid mid_deep all \
  --seed 3106 \
  --max_epochs 8 \
  --batch_size 60 \
  --output_dir rebuttal/results/p1/fusion_selected_layers
```

Required quantitative over-smoothing analysis:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/analyze_feature_oversmoothing.py \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --crowd_json /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_ge4.json \
  --backbones dinov2_vitb16 dinov3_vitb16 \
  --layers shallow mid deep last \
  --metrics crowd_token_cosine inter_person_boundary_separability foreground_background_contrast \
  --max_samples 1000 \
  --batch_size 32 \
  --output rebuttal/results/p1/feature_oversmoothing.json
```

Optional feature probe:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/analyze_feature_oversmoothing.py \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --crowd_json /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_ge4.json \
  --backbones dinov2_vitb16 dinov3_vitb16 \
  --layers shallow mid deep last \
  --metrics layerwise_probe effective_rank \
  --max_samples 2000 \
  --probe_epochs 5 \
  --batch_size 32 \
  --output rebuttal/results/p1/feature_oversmoothing_optional_probe.json
```

Required statistical reliability support; run if server budget permits:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/run_rebuttal_ablation.py \
  --group reliability \
  --dataset vat \
  --data_path /newhome/fb/dataset/videoattentiontarget \
  --crowd_json /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_ge4.json \
  --init_ckpt ./checkpoints/gazelle_dinov3_vitb16.pt \
  --variants baseline_full gazespot_full \
  --seeds 3106 3107 3108 \
  --max_epochs 8 \
  --batch_size 60 \
  --output_dir rebuttal/results/p1/reliability
```

Summarize all P1 outputs:

```bash
/home/fb/anaconda3/envs/py310/bin/python scripts/summarize_p1_results.py \
  --input_dir rebuttal/results/p1 \
  --output_md rebuttal/results/p1/p1_tables.md \
  --output_csv rebuttal/results/p1/p1_plot_data.csv
```

## Required vs Optional Analyses

Required:

- Spatial-prior controls: `none`, `fixed_gaussian`, `coordconv`, `ggsf`.
- Fusion controls: `raw_concat`, `equal_weight`, `fpn`, `sasa`.
- Feature analysis: DINOv2 vs DINOv3 at shallow/mid/deep/last layers with `crowd_token_cosine`, `inter_person_boundary_separability`, and `foreground_background_contrast` where annotations permit.
- Result templates and structured outputs with placeholders until server results exist.
- Command support for 3-seed reliability and execution of the key baseline vs full GazeSpot seed sweep if runtime permits.

Optional if runtime is too high:

- Fixed sector/cone mask.
- Full selected-layer grid beyond shallow-only, mid-only, deep/last-only, and all-layer control.
- Layer-wise probe, effective-rank analysis, PCA plot refresh, GazeFollow repeats, and multi-seed runs for every ablation variant.

## Rebuttal Table and Plot Templates

Spatial-prior controls:

| Dataset/Split | Variant | Spatial Prior | Fusion | Seed(s) | AUC ↑ | L2 ↓ | In/Out AP ↑ | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| VAT Crowd >=4 | No spatial gate | none | SASA | 3106 | TBD | TBD | TBD | required |
| VAT Crowd >=4 | Fixed Gaussian | fixed_gaussian | SASA | 3106 | TBD | TBD | TBD | required |
| VAT Crowd >=4 | CoordConv | coordconv | SASA | 3106 | TBD | TBD | TBD | required |
| VAT Crowd >=4 | Fixed sector | fixed_sector | SASA | 3106 | TBD | TBD | TBD | optional |
| VAT Crowd >=4 | GazeSpot | GGSF | SASA | 3106 | TBD | TBD | TBD | required |

Fusion controls:

| Dataset/Split | Variant | Spatial Prior | Fusion | Seed(s) | AUC ↑ | L2 ↓ | In/Out AP ↑ | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| VAT Crowd >=4 | Raw concat | GGSF | raw_concat | 3106 | TBD | TBD | TBD | required |
| VAT Crowd >=4 | Equal weight | GGSF | equal_weight | 3106 | TBD | TBD | TBD | required |
| VAT Crowd >=4 | FPN-style | GGSF | fpn | 3106 | TBD | TBD | TBD | required |
| VAT Crowd >=4 | Selected shallow/mid/deep | GGSF | selected_layers | 3106 | TBD | TBD | TBD | optional |
| VAT Crowd >=4 | GazeSpot | GGSF | SASA | 3106 | TBD | TBD | TBD | required |

Feature over-smoothing metrics:

| Backbone | Layer | Crowd Token Cosine ↓ | Inter-Person Boundary Separability ↑ | Foreground/Background Contrast ↑ | Samples | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| DINOv2 ViT-B/16 | shallow | TBD | TBD | TBD | TBD | required |
| DINOv2 ViT-B/16 | mid | TBD | TBD | TBD | TBD | required |
| DINOv2 ViT-B/16 | deep | TBD | TBD | TBD | TBD | required |
| DINOv2 ViT-B/16 | last | TBD | TBD | TBD | TBD | required |
| DINOv3 ViT-B/16 | shallow | TBD | TBD | TBD | TBD | required |
| DINOv3 ViT-B/16 | mid | TBD | TBD | TBD | TBD | required |
| DINOv3 ViT-B/16 | deep | TBD | TBD | TBD | TBD | required |
| DINOv3 ViT-B/16 | last | TBD | TBD | TBD | TBD | required |

Reliability table:

| Dataset/Split | Method | Seeds | AUC mean ± std ↑ | L2 mean ± std ↓ | In/Out AP mean ± std ↑ | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| VAT Crowd >=4 | Baseline | 3106/3107/3108 | TBD | TBD | TBD | required if runtime permits |
| VAT Crowd >=4 | GazeSpot | 3106/3107/3108 | TBD | TBD | TBD | required if runtime permits |

Plot templates:

- Spatial-prior bar plot: x-axis variant, y-axis L2/AUC, grouped by required vs optional.
- Fusion bar plot: x-axis fusion strategy, y-axis L2/AUC.
- Feature line plot: x-axis layer depth, y-axis token cosine or contrast, two lines for DINOv2 and DINOv3.
- Reliability error-bar plot: x-axis method, y-axis Crowd L2, error bars show standard deviation.

## Risks / Trade-offs

- [Ablation runtime exceeds rebuttal window] → Run required single-seed VAT Crowd controls first, then optional fixed-sector, selected-layer grid, and probe analyses only if time remains.
- [Fixed sector mask is arbitrary without head pose] → Keep it optional and describe it as a hand-designed 2D sector control, not a physical gaze cone.
- [CoordConv changes channel dimensions broadly] → Isolate adapter logic in `gazelle/ablation_variants.py` and use metadata tests to ensure only the intended variant changes.
- [FPN fusion may add parameters beyond SASA] → Report trainable parameter counts from P0 helpers if available and note architecture differences in metadata.
- [Crowd JSON naming differs on the server] → Commands use `test_crowd_ge4.json`; implementation should also accept `test_crowd_gt4.json` and fail with available alternatives listed.
- [DINOv2/DINOv3 preprocessing differs] → The feature analysis script must record input size, transform, backbone checkpoint, layers, and sample count in output metadata.
