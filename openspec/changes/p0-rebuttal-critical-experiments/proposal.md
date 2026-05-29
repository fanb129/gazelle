## Why

The ACM MM reviews repeatedly ask for must-have evidence on OOD generalization, head-box noise robustness, runtime efficiency, and transparent parameter reporting. This change prepares the local code and server run plan for those P0 rebuttal experiments while leaving experimental numbers blank until they are produced on `fb@3090.lab`.

## What Changes

- Add GOO-real preprocessing and evaluation support that can run against the server dataset at `/newhome/fb/dataset/gooreal_data` without requiring local dataset access.
- Add a metrics-only robustness evaluation that compares the DINOv3 baseline and GazeSpot under deterministic head bounding box perturbations at `0%`, `5%`, `10%`, and `20%` translation/scale jitter.
- Extend `scripts/compute_flops.py` into a reproducible complexity/latency/FPS report for RTX 3090, including trainable parameters, total parameters including the frozen VFM, MACs/FLOPs, latency in ms/img, and FPS.
- Correct parameter-reporting outputs and table templates so every report distinguishes `Trainable Params` from `Total Params`.
- Produce rebuttal-ready table templates and server commands, with explicit placeholders only. Experimental values must not be invented and must be filled only after the user runs the commands on the server and provides outputs.
- Keep scope limited to P0. This change will not implement P1 ablations, feature over-smoothing analysis, statistical multi-seed studies, ChildPlay evaluation, or P2 rebuttal prose except for result table interfaces needed by P0.

## Capabilities

### New Capabilities

- `gooreal-evaluation`: Preprocess and evaluate GOO-real using the existing GazeSpot/baseline model interface and emit table-ready metrics.
- `bbox-noise-robustness`: Evaluate baseline vs GazeSpot under controlled head bounding box perturbation levels on VAT/Crowd and optionally GazeFollow.
- `efficiency-reporting`: Measure model complexity and deployment efficiency with trainable params, total params, MACs/FLOPs, latency, and FPS.
- `parameter-reporting`: Standardize experiment outputs and table templates so trainable and total parameters are reported separately.

### Modified Capabilities

- None.

## Impact

- Expected script changes: `scripts/compute_flops.py`, new evaluation/preprocessing scripts under `scripts/` and/or `data_prep/`, and a P0 command/template note under `rebuttal/`.
- Expected shared utility changes: bbox perturbation helpers and table/report serialization helpers, preferably under `gazelle/utils.py` or a new small utility module if that keeps evaluation scripts cleaner.
- Expected server interaction: the user pulls the branch on `fb@3090.lab`, runs the proposed commands against `/newhome/fb/dataset/gooreal_data` and existing checkpoint/dataset paths, then returns actual outputs for rebuttal table filling.
- No training behavior, model architecture, or dataset contents are changed by this proposal.
