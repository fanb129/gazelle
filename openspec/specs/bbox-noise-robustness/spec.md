# bbox-noise-robustness Specification

## Purpose
TBD - created by archiving change p0-rebuttal-critical-experiments. Update Purpose after archive.
## Requirements
### Requirement: Deterministic head-box perturbation
The system SHALL provide deterministic head bounding box perturbation for normalized boxes at jitter levels `0%`, `5%`, `10%`, and `20%`, including translation and scale jitter.

#### Scenario: Apply reproducible perturbations
- **WHEN** the user runs bbox-noise evaluation with a fixed seed and jitter levels `0 5 10 20`
- **THEN** the same perturbed boxes and metrics are produced for repeated runs with the same inputs

#### Scenario: Preserve valid normalized boxes
- **WHEN** perturbation is applied to any valid normalized head box
- **THEN** the resulting box remains clipped to `[0, 1]`, preserves `xmin < xmax` and `ymin < ymax`, and respects a minimum size guard

#### Scenario: Zero jitter is the clean baseline
- **WHEN** jitter level is `0%`
- **THEN** the evaluation uses the original boxes without perturbation

### Requirement: Robustness evaluation compares baseline and GazeSpot
The system SHALL evaluate both baseline and GazeSpot on the same perturbed examples for every requested jitter level.

#### Scenario: Evaluate VAT Crowd robustness
- **WHEN** the user runs bbox-noise evaluation on the VAT Crowd JSON
- **THEN** the command outputs per-jitter metrics for baseline and GazeSpot, including AUC, L2, and In/Out AP where applicable

#### Scenario: Optionally evaluate GazeFollow robustness
- **WHEN** the user runs bbox-noise evaluation with `--dataset gazefollow`
- **THEN** the command outputs per-jitter metrics for baseline and GazeSpot, including GazeFollow AUC, Avg L2, and Min L2

### Requirement: Robustness outputs include metadata
The system SHALL write metadata needed to interpret the robustness results.

#### Scenario: Emit perturbation metadata
- **WHEN** bbox-noise evaluation finishes
- **THEN** the structured output records dataset, split/json path, checkpoints, seed, jitter levels, perturbation definition, sample count, and metric names

### Requirement: Robustness result table is rebuttal-ready
The system SHALL include a rebuttal table template for head-box noise robustness with one row per method and jitter level.

#### Scenario: Create robustness table template
- **WHEN** the P0 command/template note is created
- **THEN** it includes rows for `0%`, `5%`, `10%`, and `20%` jitter for both baseline and GazeSpot with numeric cells marked `TBD`

