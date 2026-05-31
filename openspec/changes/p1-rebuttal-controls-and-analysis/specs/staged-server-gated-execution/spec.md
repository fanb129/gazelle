## ADDED Requirements

### Requirement: Section-gated implementation
The system SHALL treat each numbered task section as an independently delivered work package that must pass local verification and user-run server validation before the next section begins.

#### Scenario: Complete one section locally
- **WHEN** a subagent completes all code and documentation changes for one task section
- **THEN** it reports the changed files, local verification commands, local verification results, and exact server commands for that section

#### Scenario: Wait for server feedback
- **WHEN** a section has been completed locally but the user has not yet confirmed server results from `fb@3090.lab`
- **THEN** implementation of later sections remains blocked

### Requirement: Server result gate
The system SHALL require the user to review section-specific server logs, result files, or experiment metrics before later sections are started.

#### Scenario: Server results are acceptable
- **WHEN** the user confirms that a section's server run completed and the results are acceptable
- **THEN** the next task section may begin

#### Scenario: Server results fail or look suspicious
- **WHEN** the user provides failed server logs, missing outputs, or suspicious metrics for a section
- **THEN** the next action is to fix or adjust that same section rather than starting a later section

### Requirement: Section-specific artifacts
The system SHALL keep commands, outputs, and summaries separated by task section under `rebuttal/results/p1/` so server feedback can be traced to the section that produced it.

#### Scenario: Write section output
- **WHEN** a section command writes rebuttal outputs
- **THEN** output paths include a section or group-specific directory such as `variant_config`, `spatial_prior`, `fusion`, `feature_oversmoothing`, `reliability`, or `summary`
