## ADDED Requirements

### Requirement: Complexity reporting includes trainable and total parameters
The system SHALL report both trainable parameters and total parameters including the frozen VFM for every measured model variant.

#### Scenario: Count model parameters
- **WHEN** the complexity command runs for baseline and GazeSpot variants
- **THEN** each result row contains separate `Trainable Params` and `Total Params` fields

### Requirement: Complexity reporting includes MACs or FLOPs
The system SHALL report model compute using the existing `scripts/compute_flops.py` path, with explicit labeling of MACs and/or FLOPs.

#### Scenario: Measure model compute
- **WHEN** the complexity command profiles a model with a dummy image and bbox input
- **THEN** the structured output records the measured compute value and the profiling tool or method used

### Requirement: Runtime reporting includes latency and FPS
The system SHALL benchmark inference latency and FPS on CUDA with warmup iterations, measured iterations, synchronization, and device metadata.

#### Scenario: Measure RTX 3090 runtime
- **WHEN** the user runs the complexity command on `fb@3090.lab` with `--device cuda`
- **THEN** the output includes latency in ms/img, FPS, batch size, warmup iterations, measured iterations, and CUDA device name

#### Scenario: CUDA unavailable
- **WHEN** the user requests CUDA runtime measurement but CUDA is unavailable
- **THEN** the command fails with a clear error instead of reporting CPU numbers as RTX 3090 results

### Requirement: Complexity result table is rebuttal-ready
The system SHALL include a rebuttal table template for baseline and GazeSpot complexity/runtime results.

#### Scenario: Create complexity table template
- **WHEN** the P0 command/template note is created
- **THEN** it includes columns for method, input size, trainable params, total params, MACs/FLOPs, latency ms/img, FPS, and device with numeric cells marked `TBD`
