## ADDED Requirements

### Requirement: Fusion ablation variants
The system SHALL provide opt-in fusion variants for P1 rebuttal ablations, including `raw_concat`, `equal_weight`, `fpn`, `sasa`, and `selected_layers` when feasible.

#### Scenario: Configure required fusion controls
- **WHEN** a rebuttal training or evaluation command is run with `--fusion raw_concat`, `--fusion equal_weight`, `--fusion fpn`, or `--fusion sasa`
- **THEN** the model uses exactly the requested fusion behavior and records the fusion mode in output metadata

#### Scenario: Configure selected-layer fusion
- **WHEN** a rebuttal command is run with `--fusion selected_layers` and a selected-layer set
- **THEN** the model uses only the requested DINO feature layers and records the resolved layer indices or named preset in output metadata

### Requirement: Fair fusion comparison
The system SHALL train and evaluate fusion variants under the same backbone, decoder, optimizer settings, dataset split, seed, and epoch budget unless an output metadata field explicitly documents an intentional difference.

#### Scenario: Compare fusion variants on VAT Crowd dense
- **WHEN** the required fusion ablation group is run on VAT Crowd dense
- **THEN** each required variant is trained from the same initialization and evaluated on the same Crowd dense JSON path with identical metrics

### Requirement: FPN-style fusion control
The system SHALL provide an FPN-style multi-layer fusion control that uses lateral projection and top-down aggregation instead of SASA's learned per-scale softmax routing.

#### Scenario: Run FPN without SASA routing
- **WHEN** the `fpn` fusion variant is selected
- **THEN** the model uses FPN-style aggregation and does not use SASA scale-attention weights for that run

### Requirement: Fusion result outputs
The system SHALL emit structured fusion ablation outputs that can populate rebuttal tables and plot data.

#### Scenario: Write table-ready fusion ablation results
- **WHEN** the fusion ablation command completes
- **THEN** it writes JSON and CSV or Markdown-ready rows containing dataset split, variant, spatial prior, fusion strategy, selected layers when applicable, seed, checkpoint path, AUC, L2, In/Out AP when applicable, and sample count

#### Scenario: Mark optional selected-layer grid
- **WHEN** selected-layer variants are not run because runtime is too high
- **THEN** the result summary marks selected-layer variants as optional/not run rather than filling metric cells
