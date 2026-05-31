## ADDED Requirements

### Requirement: Quantitative feature over-smoothing metrics
The system SHALL provide a metrics-only analysis command that computes quantitative feature smoothness and separability metrics for DINOv2 and DINOv3 across shallow, mid, deep, and last layers on crowded examples.

#### Scenario: Compute required feature metrics
- **WHEN** the feature over-smoothing analysis command is run with required metrics
- **THEN** it computes `crowd_token_cosine`, `inter_person_boundary_separability`, and `foreground_background_contrast` where the required annotations are available

#### Scenario: Report unavailable metric coverage
- **WHEN** a required metric cannot be computed for some samples because annotations or valid regions are unavailable
- **THEN** the output records the skipped sample count and reason instead of fabricating the metric

### Requirement: DINOv2 and DINOv3 layer comparison
The system SHALL compare DINOv2 and DINOv3 features using matched layer-depth labels and record the exact resolved backbone names, checkpoint paths, transforms, input sizes, layer indices, and sample counts.

#### Scenario: Analyze matched layers
- **WHEN** the analysis command is run with `--backbones dinov2_vitb16 dinov3_vitb16` and `--layers shallow mid deep last`
- **THEN** the output contains one row per backbone and layer label with metric values or explicit not-computed reasons

### Requirement: Optional heavier probes
The system SHALL support optional heavier feature analyses such as layer-wise probing, effective rank, or refreshed PCA plot data without making them required for the P1 rebuttal path.

#### Scenario: Run optional probe metric
- **WHEN** the analysis command includes `layerwise_probe` or `effective_rank`
- **THEN** the output marks those metrics as optional and records probe settings such as train/validation split, epochs, and sample count

### Requirement: Feature analysis result templates
The system SHALL provide rebuttal-ready tables and plot data templates for feature over-smoothing analysis with placeholders until server-produced values exist.

#### Scenario: Generate layer-wise table template
- **WHEN** P1 command documentation is created
- **THEN** it includes a table template with rows for DINOv2 and DINOv3 shallow/mid/deep/last layers and columns for token cosine, boundary separability, foreground/background contrast, sample count, and notes

#### Scenario: Generate plot-ready trend data
- **WHEN** feature analysis outputs are summarized
- **THEN** the summary includes plot-ready CSV rows with backbone, layer order, metric name, metric value, sample count, and optional confidence/statistical fields if available
