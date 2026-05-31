## ADDED Requirements

### Requirement: Seed sweep execution support
The system SHALL provide command support for running key rebuttal comparisons across multiple seeds and summarizing mean and standard deviation.

#### Scenario: Run three-seed key comparison
- **WHEN** the reliability command is run with seeds `3106 3107 3108` for baseline and full GazeSpot on VAT Crowd dense
- **THEN** it trains or evaluates each requested seed, writes per-seed result files, and writes an aggregate result with mean and standard deviation for each metric

#### Scenario: Runtime-limited reliability fallback
- **WHEN** the user does not run all requested seeds because server runtime is too high
- **THEN** the summary marks the reliability table as incomplete and does not present single-seed values as mean/std evidence

### Requirement: Reliability metric aggregation
The system SHALL aggregate reliability results with explicit method names, seeds, dataset split, sample count, checkpoint paths, metric means, metric standard deviations, and missing-seed diagnostics.

#### Scenario: Aggregate complete seeds
- **WHEN** all expected seed outputs exist for a method
- **THEN** the summary reports `mean ± std` for AUC, L2, and In/Out AP when applicable

#### Scenario: Detect missing seed output
- **WHEN** one or more expected seed outputs are absent or malformed
- **THEN** the summary lists the missing seed identifiers and keeps aggregate metric cells as `TBD` or `incomplete`

### Requirement: Reliability rebuttal templates
The system SHALL include a rebuttal-ready reliability table template for the key Crowd dense subset comparison.

#### Scenario: Create reliability table template
- **WHEN** P1 command documentation is created
- **THEN** it includes rows for baseline and GazeSpot with seed list, AUC mean/std, L2 mean/std, In/Out AP mean/std when applicable, and notes
