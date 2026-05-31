## Execution Protocol: Section-Gated Server Validation

- [ ] 0.1 Treat each numbered section below as an independent subagent work package.
- [ ] 0.2 A subagent MUST complete only its assigned section unless the user explicitly approves continuing.
- [ ] 0.3 After each section, stop and report changed files, local verification commands/results, and exact server commands.
- [ ] 0.4 The user syncs code to `fb@3090.lab`, runs the section-specific server command(s), and returns logs/result files.
- [ ] 0.5 Do not start the next section until the user confirms the previous section's server result is acceptable or gives a fix request for that same section.
- [ ] 0.6 Keep result outputs separated under `rebuttal/results/p1/<section_or_group>/`.

## 1. Variant Configuration and Metadata

- [x] 1.1 Add a small variant helper module, preferably `gazelle/ablation_variants.py`, for parsing `--spatial_prior`, `--fusion`, `--selected_layers`, and seed metadata.
- [x] 1.2 Preserve existing `--use_sasa` and `--use_ggsf` behavior while mapping new rebuttal modes onto the current production defaults.
- [x] 1.3 Add metadata serialization so every P1 run records dataset split, backbone, input size, seed, spatial prior, fusion strategy, selected layers, checkpoint path, and sample count.
- [x] 1.4 Add a `--print_plan_only` or equivalent dry-run path that validates variant arguments and writes/prints the resolved run plan without training.
- [x] 1.5 Local verification: run parser/unit smoke checks for mode combinations and metadata serialization without requiring full server datasets.
- [ ] 1.6 Server gate: after sync, run the Section 1 dry-run command from `design.md`; proceed only after the user confirms the resolved plan is correct.

## 2. Spatial-Prior Controls

- [ ] 2.1 Implement `spatial_prior=none` as an all-ones/no-gate control.
- [ ] 2.2 Implement `spatial_prior=fixed_gaussian` as a deterministic head-centered mask with documented sigma derived from normalized head box size.
- [ ] 2.3 Implement `spatial_prior=coordconv` as coordinate/head-relative conditioning without learned multiplicative GGSF masking.
- [ ] 2.4 Keep `spatial_prior=ggsf` mapped to the existing learned GGSF module.
- [ ] 2.5 Implement `spatial_prior=fixed_sector` only if it can be kept deterministic and honestly described as a 2D sector control; otherwise make the command fail clearly and mark the analysis optional.
- [ ] 2.6 Add unit or smoke checks that fixed spatial masks are deterministic for the same bbox and feature-map dimensions.
- [ ] 2.7 Local verification: run mask determinism tests, shape checks, and `python -m py_compile` for modified Python files.
- [ ] 2.8 Server gate: after sync, run the required spatial-prior ablation command from `design.md`; proceed only after the user confirms outputs under `rebuttal/results/p1/spatial_prior/` are structurally valid and metrics look sane.

## 3. Fusion Controls

- [ ] 3.1 Implement `fusion=raw_concat` as the existing multi-layer concatenate plus projection baseline.
- [ ] 3.2 Implement `fusion=equal_weight` by applying equal per-layer weights before concatenation/projection.
- [ ] 3.3 Implement `fusion=sasa` using the existing learned SASA scale-attention path.
- [ ] 3.4 Implement `fusion=fpn` with lateral projection and top-down feature aggregation, without SASA softmax routing.
- [ ] 3.5 Implement `fusion=selected_layers` with named presets for shallow, mid, deep/last, shallow_mid, mid_deep, and all where feasible.
- [ ] 3.6 Add shape and metadata smoke checks for every required fusion mode.
- [ ] 3.7 Local verification: run fusion shape checks, selected-layer parsing checks, and `python -m py_compile` for modified Python files.
- [ ] 3.8 Server gate: after sync, run the required fusion ablation command from `design.md`; proceed only after the user confirms outputs under `rebuttal/results/p1/fusion/` are structurally valid and metrics look sane.

## 4. Training and Evaluation Runner

- [ ] 4.1 Extend `scripts/train_vat.py` and `scripts/train_gazefollow.py` with P1 variant arguments and explicit seed handling.
- [ ] 4.2 Extend `scripts/eval_vat.py` and `scripts/eval_gazefollow.py` or add a metrics-only path so named variant checkpoints can be evaluated without relying on visualization outputs.
- [ ] 4.3 Create `scripts/run_rebuttal_ablation.py` to run required spatial-prior and fusion groups, save per-run commands/manifests, and emit JSON/CSV metrics under `rebuttal/results/p1/`.
- [ ] 4.4 Ensure VAT Crowd dense is the required evaluation target and GazeFollow is optional unless the user asks for it or server time remains.
- [ ] 4.5 Ensure commands accept both `test_crowd_ge4.json` and `test_crowd_gt4.json` style dense-subset paths, failing with available alternatives when neither exists.
- [ ] 4.6 Local verification: run dry-run/plan-only commands for each runner group and compile all modified scripts.
- [ ] 4.7 Server gate: after sync, run a short runner smoke command first, then the smallest full command needed by Sections 2 or 3; proceed only after the user confirms checkpoints, manifests, and metric files are generated as expected.

## 5. Quantitative Feature Over-Smoothing Analysis

- [ ] 5.1 Create `scripts/analyze_feature_oversmoothing.py` to extract DINOv2 and DINOv3 features at shallow, mid, deep, and last layers on Crowd dense examples.
- [ ] 5.2 Implement required metric `crowd_token_cosine` for crowded person/head regions.
- [ ] 5.3 Implement required metric `inter_person_boundary_separability` when multiple annotated head/person regions and valid boundary bands are available.
- [ ] 5.4 Implement required metric `foreground_background_contrast` using valid person/head and background token sets.
- [ ] 5.5 Record skipped sample counts and reasons for metrics that cannot be computed on particular examples.
- [ ] 5.6 Add optional support for `layerwise_probe`, `effective_rank`, and PCA plot data if runtime permits.
- [ ] 5.7 Local verification: run argument-validation and small synthetic-region metric tests without requiring the full VAT dataset.
- [ ] 5.8 Server gate: after sync, run the required quantitative over-smoothing command from `design.md`; proceed only after the user confirms `feature_oversmoothing.json` has complete required rows or clear skipped-sample reasons.

## 6. Statistical Reliability

- [ ] 6.1 Add seed-sweep support to `scripts/run_rebuttal_ablation.py` for seeds `3106`, `3107`, and `3108`.
- [ ] 6.2 Implement aggregation that reports per-method mean and standard deviation for AUC, L2, and In/Out AP when all expected seed outputs exist.
- [ ] 6.3 Mark aggregate cells as `TBD` or `incomplete` when one or more seed outputs are missing.
- [ ] 6.4 Prioritize the required-if-runtime-permits comparison: baseline DINOv3 last-layer or raw-concat baseline vs full GazeSpot on VAT Crowd dense.
- [ ] 6.5 Local verification: run aggregation tests with synthetic per-seed JSON files, including complete and missing-seed cases.
- [ ] 6.6 Server gate: after sync, run the reliability command only if runtime budget permits; proceed only after the user confirms complete mean/std outputs or explicitly accepts an incomplete/runtime-limited result.

## 7. Rebuttal Result Package

- [ ] 7.1 Create `rebuttal/p1_controls_and_analysis.md` with required and optional analysis lists, exact server commands from `design.md`, section-gated workflow instructions, and a no-invented-numbers warning.
- [ ] 7.2 Add table templates for spatial-prior controls, fusion controls, feature over-smoothing metrics, and reliability mean/std.
- [ ] 7.3 Add plot-data templates for spatial-prior bars, fusion bars, DINOv2/DINOv3 layer-wise feature trends, and reliability error bars.
- [ ] 7.4 Create `scripts/summarize_p1_results.py` to convert P1 JSON outputs into Markdown tables and plot-ready CSV files.
- [ ] 7.5 Keep all numeric cells as `TBD` until server outputs are produced by the user.
- [ ] 7.6 Local verification: run summary generation against synthetic P1 JSON files and confirm Markdown/CSV outputs keep missing numbers as `TBD`.
- [ ] 7.7 Server gate: after sync and after prior result files exist, run the summary command from `design.md`; finish only after the user confirms `p1_tables.md` and `p1_plot_data.csv` are usable by P2.

## 8. Expected Changed Files

- [ ] 8.1 Update `gazelle/model.py` for DINOv3 variant plumbing.
- [ ] 8.2 Update `gazelle/model_dinov2.py` only where needed for DINOv2 feature analysis or mirrored variant plumbing.
- [ ] 8.3 Add `gazelle/ablation_variants.py`.
- [ ] 8.4 Update `scripts/train_gazefollow.py`, `scripts/train_vat.py`, `scripts/eval_gazefollow.py`, and `scripts/eval_vat.py`.
- [ ] 8.5 Add `scripts/run_rebuttal_ablation.py`, `scripts/analyze_feature_oversmoothing.py`, and `scripts/summarize_p1_results.py`.
- [ ] 8.6 Add `rebuttal/p1_controls_and_analysis.md` and generated outputs under `rebuttal/results/p1/` after server runs.
- [ ] 8.7 Treat this section as an inventory checklist, not a standalone implementation package; update it as each gated section lands.

## 9. Final Acceptance Criteria and Verification

- [ ] 9.1 Required spatial-prior controls include `none`, `fixed_gaussian`, `coordconv`, and `ggsf`; `fixed_sector` is explicitly marked optional if not feasible.
- [ ] 9.2 Required fusion controls include `raw_concat`, `equal_weight`, `fpn`, and `sasa`; selected-layer variants are optional if runtime is too high.
- [ ] 9.3 Required feature analysis compares DINOv2 vs DINOv3 at shallow, mid, deep, and last layers with token cosine, boundary separability, and foreground/background contrast where annotations permit.
- [ ] 9.4 Statistical reliability supports 3-seed mean/std and does not present incomplete single-seed runs as multi-seed evidence.
- [ ] 9.5 P1 artifacts do not implement or duplicate P0 GOO-real, bbox-noise robustness, latency/FPS, or parameter-reporting work.
- [ ] 9.6 P1 artifacts do not write final rebuttal text; they provide tables, plots, and command templates for P2 to consume.
- [ ] 9.7 Run `python -m py_compile` on new or modified Python scripts after each implementation section, not only at the end.
- [ ] 9.8 Run local smoke tests for the current section before asking the user to sync code to the server.
- [ ] 9.9 Confirm each section's server gate before starting the next numbered section.
- [ ] 9.10 Run `openspec status --change p1-rebuttal-controls-and-analysis` and confirm the change is apply-ready.
