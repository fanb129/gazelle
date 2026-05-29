## 1. GOO-real Evaluation Support

- [x] 1.1 Inspect GOO-real server files and document discovered pickle/archive keys without assuming local dataset access.
- [x] 1.2 Implement `data_prep/preprocess_gooreal.py` to convert GOO-real into the normalized evaluation JSON schema used by existing GazeFollow-style evaluators.
- [x] 1.3 Implement `scripts/eval_gooreal.py` to load baseline and GazeSpot checkpoints, evaluate the same GOO-real examples, and emit JSON/CSV/Markdown-ready metrics.
- [x] 1.4 Add clear failure messages for missing images, unexpected annotation keys, invalid boxes, or missing gaze targets.
- [x] 1.5 Verify GOO-real preprocessing/evaluation commands can run in dry-run or schema-check mode locally without requiring the real dataset.

## 2. Head Bounding Box Noise Robustness

- [ ] 2.1 Add a deterministic bbox perturbation helper for normalized boxes with translation and scale jitter at `0%`, `5%`, `10%`, and `20%`.
- [ ] 2.2 Implement `scripts/eval_bbox_noise.py` for VAT/VAT-Crowd, comparing baseline and GazeSpot for every jitter level.
- [ ] 2.3 Support optional GazeFollow evaluation in `scripts/eval_bbox_noise.py` using the same perturbation and metadata path.
- [ ] 2.4 Ensure the `0%` jitter path uses original boxes and can reproduce the clean evaluation metrics for the same split/checkpoints.
- [ ] 2.5 Emit structured metadata with dataset, split/json path, checkpoints, seed, jitter levels, perturbation definition, sample counts, and metric names.

## 3. Latency, FPS, and Complexity Reporting

- [ ] 3.1 Extend `scripts/compute_flops.py` with CLI arguments for device, batch size, warmup iterations, measured iterations, and output path.
- [ ] 3.2 Replace rebuttal-facing `Head Params`/generic `Params` labels with separate `Trainable Params` and `Total Params` fields.
- [ ] 3.3 Report total parameters including the frozen VFM, trainable parameters, optional diagnostic non-backbone parameters, MACs/FLOPs, latency ms/img, FPS, input size, batch size, and device name.
- [ ] 3.4 Add CUDA synchronization around measured iterations and fail clearly if RTX/CUDA timing is requested but CUDA is unavailable.
- [ ] 3.5 Emit machine-readable JSON plus a Markdown/CSV-friendly summary table.

## 4. P0 Rebuttal Command and Table Package

- [ ] 4.1 Create `rebuttal/p0_experiment_commands.md` with the server commands from `design.md`, including `fb@3090.lab`, `v1`, `/newhome/fb/dataset/gooreal_data`, checkpoint args, and output paths.
- [ ] 4.2 Add GOO-real, bbox-noise robustness, and complexity/runtime table templates with `TBD` in all numeric cells.
- [ ] 4.3 Add a prominent note that experimental numbers must not be invented and must be filled only after the user runs commands on the server.
- [ ] 4.4 Keep the P0 note limited to interfaces and result placeholders; do not add P1 ablation plans or P2 final rebuttal prose.

## 5. Verification

- [ ] 5.1 Run `python -m py_compile` on all new or modified Python scripts.
- [ ] 5.2 Run local schema-check/dry-run commands for GOO-real preprocessing and bbox-noise evaluation without needing the server dataset.
- [ ] 5.3 Run `scripts/compute_flops.py` in CPU smoke-test mode locally only to validate CLI/output formatting, without claiming RTX 3090 latency/FPS.
- [ ] 5.4 Run `openspec status --change p0-rebuttal-critical-experiments` and confirm the change remains apply-ready.

## Acceptance Criteria

- GOO-real support is prepared as scripts and commands, but no local run assumes the dataset exists outside `/newhome/fb/dataset/gooreal_data` on `fb@3090.lab`.
- VAT/Crowd bbox-noise robustness is required; optional GazeFollow robustness is available through the same script interface.
- Perturbation levels include exactly `0%`, `5%`, `10%`, and `20%` unless the user explicitly requests more.
- Complexity reporting includes `Trainable Params`, `Total Params`, MACs/FLOPs, latency ms/img, FPS, input size, batch size, and device.
- Rebuttal table templates contain placeholders only until real server outputs are returned by the user.
- No P1 ablations, P2 writing, or ChildPlay evaluation are implemented under this change.

## Expected Changed Files

- `data_prep/preprocess_gooreal.py`
- `scripts/eval_gooreal.py`
- `scripts/eval_bbox_noise.py`
- `scripts/compute_flops.py`
- `gazelle/utils.py` or `gazelle/eval_utils.py`
- `rebuttal/p0_experiment_commands.md`

## Required Server Commands

Use the full commands in `design.md` and mirror them in `rebuttal/p0_experiment_commands.md` during apply. Required command groups are:

- GOO-real preprocessing on `/newhome/fb/dataset/gooreal_data`
- GOO-real baseline vs GazeSpot evaluation
- VAT Crowd bbox-noise robustness at `0 5 10 20`
- RTX 3090 complexity/latency/FPS measurement from `scripts/compute_flops.py`

Optional command group:

- GazeFollow bbox-noise robustness at `0 5 10 20`
