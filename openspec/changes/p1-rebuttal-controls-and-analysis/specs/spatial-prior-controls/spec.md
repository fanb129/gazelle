## ADDED Requirements

### Requirement: Spatial prior ablation variants
The system SHALL provide opt-in spatial-prior variants for P1 rebuttal ablations, including `none`, `fixed_gaussian`, `coordconv`, `ggsf`, and `fixed_sector` when feasible.

#### Scenario: Configure required spatial priors
- **WHEN** a rebuttal training or evaluation command is run with `--spatial_prior none`, `--spatial_prior fixed_gaussian`, `--spatial_prior coordconv`, or `--spatial_prior ggsf`
- **THEN** the model uses exactly the requested spatial-prior behavior and records that behavior in output metadata

#### Scenario: Configure optional fixed sector prior
- **WHEN** a rebuttal command is run with `--spatial_prior fixed_sector`
- **THEN** the model either runs a documented deterministic 2D sector mask or fails clearly that the fixed sector control is infeasible without head-pose or gaze-direction cues

### Requirement: Fixed spatial masks are deterministic
The system SHALL generate fixed Gaussian and fixed sector masks deterministically from normalized head bounding boxes and feature-map dimensions.

#### Scenario: Reproduce fixed Gaussian mask
- **WHEN** the same image, normalized head box, feature-map size, and Gaussian configuration are used twice
- **THEN** the generated fixed Gaussian mask values are identical and the output metadata includes the Gaussian sigma rule

#### Scenario: Avoid learned parameters in fixed masks
- **WHEN** a fixed spatial-prior variant is selected
- **THEN** the spatial-prior component has no learned mask parameters and does not instantiate the learned GGSF MLP

### Requirement: CoordConv spatial control
The system SHALL provide a CoordConv-style coordinate conditioning control that exposes spatial coordinates and head-relative coordinates without using learned multiplicative GGSF masking.

#### Scenario: Use coordinate channels without GGSF gate
- **WHEN** the `coordconv` spatial-prior variant is selected
- **THEN** the model receives coordinate/head-relative channels or their projected equivalent and does not multiply feature maps by a learned GGSF mask

### Requirement: Spatial prior result outputs
The system SHALL emit structured spatial-prior ablation outputs that compare required variants on the same dataset split, seed, backbone, training budget, and evaluation metrics.

#### Scenario: Write table-ready spatial ablation results
- **WHEN** the spatial-prior ablation command completes
- **THEN** it writes JSON and CSV or Markdown-ready rows containing dataset split, variant, spatial prior, fusion strategy, seed, checkpoint path, AUC, L2, In/Out AP when applicable, and sample count

#### Scenario: Preserve numerical honesty before server runs
- **WHEN** server outputs do not exist yet
- **THEN** rebuttal table templates retain `TBD` placeholders and contain no invented or estimated metric values
