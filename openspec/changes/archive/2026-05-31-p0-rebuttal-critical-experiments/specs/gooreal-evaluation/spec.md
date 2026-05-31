## ADDED Requirements

### Requirement: GOO-real preprocessing emits existing evaluation schema
The system SHALL provide a GOO-real preprocessing command that reads the server dataset at `/newhome/fb/dataset/gooreal_data` and writes a JSON file compatible with the existing GazeFollow-style model/evaluation input schema.

#### Scenario: Convert server GOO-real annotations
- **WHEN** the user runs the GOO-real preprocessing command on `fb@3090.lab`
- **THEN** the command writes a JSON file containing image `path`, `width`, `height`, and per-head `bbox_norm`, `gazex_norm`, `gazey_norm`, and `inout` fields

#### Scenario: Missing or unexpected GOO-real structure
- **WHEN** the preprocessing command cannot find expected annotation keys or image paths
- **THEN** it fails with a clear error describing discovered files or keys instead of writing partial or fabricated annotations

### Requirement: GOO-real evaluation compares baseline and GazeSpot
The system SHALL provide a metrics-only GOO-real evaluation command that loads baseline and GazeSpot checkpoints, evaluates the same preprocessed GOO-real examples, and emits table-ready metrics.

#### Scenario: Evaluate both methods on GOO-real
- **WHEN** the user runs the GOO-real evaluation command with valid dataset JSON and checkpoints
- **THEN** the command outputs metrics for both `Baseline (DINOv3 last-layer)` and `GazeSpot` in structured JSON and/or CSV form

#### Scenario: Preserve numerical honesty
- **WHEN** GOO-real evaluation has not been run on the server
- **THEN** generated table templates retain `TBD` placeholders and do not contain invented, estimated, or copied numbers

### Requirement: GOO-real result table is rebuttal-ready
The system SHALL include a rebuttal table template for GOO-real OOD evaluation with method names, metric columns, and placeholders for server-produced values.

#### Scenario: Create OOD table template
- **WHEN** the P0 command/template note is created
- **THEN** it includes a GOO-real table with rows for baseline and GazeSpot and columns for applicable gaze metrics such as AUC, Avg L2, and Min L2
