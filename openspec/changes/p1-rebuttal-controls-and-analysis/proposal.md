## Why

Reviewer 7pTL and CoU9 question whether GazeSpot's gains come from the proposed head-conditioned spatial focus and dynamic scale routing, or from simpler coordinate/multi-scale alternatives. This change prepares the P1 rebuttal controls and quantitative claim-hardening analyses needed to support GGSF, SASA, and the DINOv3 over-smoothing motivation without duplicating the already completed P0 GOO-real, bbox-noise, latency, or parameter-reporting work.

## What Changes

- Add a rebuttal ablation suite for GGSF controls: no spatial gate, learned GGSF, fixed Gaussian head-centered mask, CoordConv-style coordinate channels, and a fixed cone/sector mask when feasible.
- Add a rebuttal ablation suite for SASA controls: raw multi-layer concatenation, equal-weight fusion, FPN-style fusion, selected-layer variants, and learned SASA.
- Add quantitative feature over-smoothing analysis for DINOv2 vs DINOv3 and shallow/mid/deep layers, with required metrics that can run on Crowd dense examples and optional heavier probes if runtime allows.
- Add statistical reliability support for the key Crowd dense subset comparison using three seeds where training budget permits, and a single-seed fallback that explicitly remains lower confidence.
- Add rebuttal-ready result table and plot templates with `TBD` placeholders only; numbers are filled only after the user runs commands on `fb@3090.lab` and returns outputs.
- Structure implementation as gated sections: after each task section is completed locally, the user syncs code to `fb@3090.lab`, runs the corresponding server commands, reviews results, and only then starts the next section.
- Keep this change out of P0 scope: no GOO-real evaluation, bbox-noise robustness, latency/FPS measurement, or final rebuttal prose except for interfaces needed by these P1 artifacts.

## Capabilities

### New Capabilities
- `spatial-prior-controls`: Train/evaluate GGSF alternatives that test whether a learned head-conditioned gate beats simpler fixed or coordinate-only spatial priors.
- `fusion-strategy-controls`: Train/evaluate SASA alternatives that test whether dynamic scale routing beats raw concatenation, equal weights, FPN-style fusion, and selected-layer baselines.
- `feature-oversmoothing-analysis`: Quantify DINOv2/DINOv3 and layer-wise feature smoothness/separability on crowded examples using feature statistics and optional probing.
- `statistical-reliability-reporting`: Run and report multi-seed mean/std or confidence intervals for key Crowd dense subset comparisons when runtime permits.
- `staged-server-gated-execution`: Require each implementation section to pass local verification and user-run server validation before the next section begins.

### Modified Capabilities
- None.

## Impact

- Expected model changes: `gazelle/model.py`, `gazelle/model_dinov2.py`, and optionally a small new module such as `gazelle/ablation_variants.py` to keep GGSF/SASA alternatives isolated from the production path.
- Expected script changes: `scripts/train_gazefollow.py`, `scripts/train_vat.py`, `scripts/eval_gazefollow.py`, `scripts/eval_vat.py`, plus new P1 scripts such as `scripts/run_rebuttal_ablation.py`, `scripts/analyze_feature_oversmoothing.py`, and `scripts/summarize_p1_results.py`.
- Expected rebuttal artifact changes: a P1 command/template note such as `rebuttal/p1_controls_and_analysis.md` and generated outputs under `rebuttal/results/p1/`.
- Expected server interaction: the user pulls the branch on `fb@3090.lab`, runs the proposed commands against existing VAT/GazeFollow/Crowd paths and checkpoints, then returns structured outputs for table filling.
- Expected collaboration flow: section-scoped local code changes, local smoke checks, user sync to server, server experiment feedback, artifact/task update, then proceed to the next section.
- No dataset contents, final rebuttal text, or P0 experiment ownership are changed by this proposal.
