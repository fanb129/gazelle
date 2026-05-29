# GazeSpot Rebuttal Strategy

Date: 2026-05-29

This document is the shared coordination plan for the rebuttal phase. It records the overall rebuttal strategy, the experiment priorities, and the exact `/opsx:propose` prompts for the three follow-up agents.

## Current Situation

Submission 3106 has 3 Weak Accept reviews and 2 Weak Reject reviews.

The paper is generally seen as well motivated and technically sound, but the negative reviews concentrate on a few repeated concerns:

- The term "physical frustum" may overclaim what GGSF actually implements.
- Generalization evidence is incomplete because GOO-real and ChildPlay are not evaluated.
- GGSF depends on head bounding box quality, but noisy box robustness is not measured.
- Efficiency claims report trainable parameters, but not latency/FPS or total computation.
- The Crowd subset is central but under-specified for reproducibility.
- Ablations do not yet rule out simpler alternatives such as Gaussian masks, CoordConv, fixed masks, or FPN-style fusion.
- The DINOv3 over-smoothing claim relies too much on PCA visualization and needs quantitative support.

The rebuttal should be evidence-first. We should not spend most of the response defending wording. Instead, we should clarify the scope, add targeted experiments, and show that the main claims survive stronger controls.

## Core Rebuttal Positioning

Recommended framing:

> We thank the reviewers for the constructive comments. We agree that the term "physical frustum" should be interpreted as a frustum-inspired, head-conditioned spatial prior rather than a full 3D gaze-cone estimator. We will revise the terminology accordingly and add new experiments on OOD generalization, head-box noise robustness, runtime efficiency, stronger ablations, and Crowd subset reproducibility.

Important wording changes for the rebuttal/final version:

- Replace strong claims like "physical frustum", "gaze cone", and "physical isolation" where they imply 3D geometry.
- Prefer "frustum-inspired spatial prior", "head-conditioned spatial gate", "relative-coordinate spatial focus", or "spotlight-style spatial mask".
- State clearly that GGSF does not use head pose, eye orientation, depth, camera geometry, or predicted gaze vectors.
- Emphasize that the contribution is a lightweight explicit spatial constraint for crowded gaze-target estimation, not a complete 3D gaze modeling module.

## Reviewer-to-Issue Map

| Reviewer | Current score | Main concerns | Response strategy |
| --- | --- | --- | --- |
| uFer | Weak Accept | GOO-real/OOD; noisy head boxes | Add GOO-real evaluation and head-box perturbation robustness. |
| fs7N | Weak Accept | 2D prior; head box dependency; smaller gains in sparse scenes | Clarify scope of GGSF, show robustness curve, explain that sparse scenes have less contextual interference so smaller gains are expected. |
| epmB | Weak Accept | Head box dependency; missing latency/FPS | Add latency/FPS/FLOPs table and noisy bbox analysis. |
| CoU9 | Weak Reject | Limited novelty; unclear GGSF learning; missing ChildPlay/GOO-real; misleading params table | Clarify GGSF mechanism, add GOO-real, rename "Params" to "Trainable Params" or report both trainable/total, add controls. |
| 7pTL | Weak Reject | Frustum overclaim; weak over-smoothing evidence; mixed gains; Crowd subset under-specified; incomplete ablations | Reduce terminology overclaim, add quantitative feature analysis, confidence intervals if possible, release details for Crowd subset, add simple-mask/FPN controls. |

## Priority Plan

### P0: Must-Have Rebuttal Evidence

P0 experiments are the highest expected score movers. They directly address the shared concerns across multiple reviewers.

1. GOO-real evaluation
   - Goal: demonstrate OOD/generalization beyond GazeFollow, VAT, and the custom Crowd subset.
   - Server context from README: GOO-real is already downloaded at `/newhome/fb/dataset/gooreal_data` on `fb@3090.lab`.
   - Expected output: preprocessing/evaluation support, commands for server execution, and a result table comparing baseline/GazeSpot.

2. Head-box noise robustness
   - Goal: measure sensitivity to inaccurate head bounding boxes.
   - Suggested perturbations: 0%, 5%, 10%, 20% jitter in translation and scale.
   - Datasets: VAT and/or GazeFollow; Crowd subset is especially valuable.
   - Expected output: robustness table or curve with AUC/L2/In-Out AP where applicable.

3. Latency/FPS/complexity
   - Goal: substantiate efficiency claims beyond trainable parameter count.
   - Starting point: `scripts/compute_flops.py`.
   - Expected output: table with trainable params, total params, FLOPs/MACs, latency ms/img, and FPS on RTX 3090.

4. Parameter table correction
   - Goal: remove the "misleading params" criticism.
   - Expected final wording: report "Trainable Params" explicitly, and optionally include "Total Params including frozen VFM" in appendix or rebuttal.

### P1: Strong Controls and Claim Hardening

P1 experiments make the paper more resilient against Reviewer 7pTL and CoU9.

1. Simple spatial-prior ablations
   - Compare GGSF against:
     - no spatial gate,
     - fixed Gaussian mask around the head,
     - CoordConv-style coordinate channels,
     - hand-designed/fixed cone or sector mask if feasible.
   - Goal: show GGSF is not merely "any coordinate conditioning".

2. Fusion strategy ablations
   - Compare SASA against:
     - raw multi-layer concatenation,
     - equal-weight fusion,
     - FPN-style fusion,
     - layer-wise selected features if feasible.
   - Goal: show dynamic scale routing matters.

3. Quantitative feature over-smoothing analysis
   - Add at least one quantitative measure:
     - token cosine similarity in crowded regions,
     - inter-person boundary separability,
     - foreground/background or person/person feature contrast,
     - layer-wise probing performance.
   - Goal: support the DINOv3 over-smoothing claim beyond PCA visualization.

4. Statistical reliability
   - If feasible, run 3 seeds for the most important comparison.
   - Minimum useful target: Crowd dense subset L2 for baseline vs GazeSpot.
   - Report mean +/- std or confidence interval.

### P2: Reproducibility, Writing, and Lower-Priority Benchmarks

P2 makes the rebuttal polished and reduces meta-review risk.

1. Crowd subset reproducibility
   - Document exact construction criteria.
   - Provide sample counts, split source, and script path.
   - Prepare release plan for sample IDs/construction script.
   - Check for confounds we can summarize: density, in-frame/out-of-frame ratio, scene distribution if feasible.

2. ChildPlay handling
   - README says ChildPlay-gaze is not downloaded because YouTube is inaccessible.
   - Do not fabricate results.
   - Best response: mention that GOO-real is added as the requested OOD benchmark; ChildPlay evaluation will be included if data access becomes available.

3. Rebuttal text assembly
   - Build a concise response organized by issue, not by reviewer only.
   - Suggested sections:
     - Clarification of GGSF/frustum wording.
     - New OOD results on GOO-real.
     - Robustness to noisy head boxes.
     - Efficiency and parameter reporting.
     - Stronger ablations and over-smoothing analysis.
     - Crowd subset reproducibility.

4. Final paper revision checklist
   - Rename/soften overclaiming terms.
   - Fix equation punctuation and formatting issues.
   - Rename "Params" column.
   - Add limitation wording that is honest but not self-defeating.

## Coordination Rules for Follow-Up Agents

- Work on branch `v1`.
- The local environment is for code/script/text changes only.
- Real training/evaluation will be run manually on the server by the user after pulling code.
- Agents must not claim experimental success until the user provides server output.
- Proposal phase and apply phase should use the same OpenSpec change id.
- After the user approves a proposal, the corresponding agent should continue into apply using the accepted proposal/design/tasks as the source of truth.
- Each agent should produce:
  - an OpenSpec proposal,
  - a design if needed,
  - tasks with clear acceptance criteria,
  - server commands to run,
  - expected result table templates,
  - a list of files it expects to change.
- Keep write scopes separated to reduce conflicts:
  - P0 agent: evaluation scripts and core experiment commands.
  - P1 agent: ablation/analysis scripts.
  - P2 agent: documentation, rebuttal draft, reproducibility notes.

## Prompt for Agent 1: P0 Proposal

Copy this into a new agent:

```text
/opsx:propose p0-rebuttal-critical-experiments

You are responsible for the P0 rebuttal work for the GazeSpot ACM MM submission. Read README.md, rebuttal/rebuttal_strategy.md, and the PDFs in rebuttal/. Create an OpenSpec proposal for the must-have rebuttal experiments.

Scope:
1. Add GOO-real evaluation support. The user has already downloaded GOO-real on the server at /newhome/fb/dataset/gooreal_data on fb@3090.lab. Local code changes should prepare preprocessing/evaluation scripts and server commands; do not assume local dataset access.
2. Add head bounding box noise robustness evaluation for baseline vs GazeSpot, preferably on VAT/Crowd and optionally GazeFollow. Include perturbation levels such as 0%, 5%, 10%, and 20% translation/scale jitter.
3. Add latency/FPS/complexity measurement, building from scripts/compute_flops.py. Report trainable params, total params including frozen VFM, FLOPs/MACs, latency ms/img, and FPS on RTX 3090.
4. Correct the parameter-reporting issue by making code/output/table templates distinguish Trainable Params from Total Params.

Proposal requirements:
- Do not implement yet. Produce OpenSpec artifacts only.
- Include exact tasks, acceptance criteria, expected changed files, and server run commands.
- Include result table templates for the rebuttal.
- Explicitly state that experimental numbers must not be invented and will be filled only after the user runs commands on the server.
- Keep scope limited to P0. Do not take over P1 ablations or P2 writing except where interfaces are needed.
```

## Prompt for Agent 2: P1 Proposal

Copy this into a new agent:

```text
/opsx:propose p1-rebuttal-controls-and-analysis

You are responsible for the P1 rebuttal work for the GazeSpot ACM MM submission. Read README.md, rebuttal/rebuttal_strategy.md, and the PDFs in rebuttal/. Create an OpenSpec proposal for stronger controls and claim-hardening analyses.

Scope:
1. Add simple spatial-prior ablations against GGSF: no spatial gate, fixed Gaussian head-centered mask, CoordConv-style coordinate channels, and a fixed cone/sector mask if feasible.
2. Add fusion strategy ablations against SASA: raw multi-layer concatenation, equal-weight fusion, FPN-style fusion, and selected-layer variants if feasible.
3. Add quantitative feature over-smoothing analysis for DINOv2 vs DINOv3 and/or shallow/mid/deep layers. Prefer metrics such as token cosine similarity in crowded regions, inter-person boundary separability, foreground/background contrast, or layer-wise probing.
4. Add statistical reliability support where feasible, such as 3-seed mean/std for the key Crowd dense subset comparison.

Proposal requirements:
- Do not implement yet. Produce OpenSpec artifacts only.
- Include exact tasks, acceptance criteria, expected changed files, and server run commands.
- Include result table/plot templates for the rebuttal.
- Explicitly state which analyses are required and which are optional if runtime is too high.
- Do not duplicate P0 work on GOO-real, bbox-noise robustness, or latency/FPS unless needed as an interface.
- Do not write the final rebuttal text; P2 owns that.
```

## Prompt for Agent 3: P2 Proposal

Copy this into a new agent:

```text
/opsx:propose p2-rebuttal-writing-and-reproducibility

You are responsible for the P2 rebuttal coordination, writing, and reproducibility work for the GazeSpot ACM MM submission. Read README.md, rebuttal/rebuttal_strategy.md, and the PDFs in rebuttal/. Create an OpenSpec proposal for the rebuttal document package and reproducibility improvements.

Scope:
1. Create a Crowd subset reproducibility note: exact construction criteria, sample counts, split source, script path, planned release artifacts, and any feasible confound checks such as density distribution and in-frame/out-of-frame ratio.
2. Prepare a rebuttal draft structure organized by issue: GGSF/frustum clarification, GOO-real OOD results placeholder, noisy bbox robustness placeholder, latency/FPS placeholder, stronger ablations placeholder, over-smoothing analysis placeholder, and Crowd subset reproducibility.
3. Prepare final-paper revision checklist: soften "physical frustum/gaze cone/physical isolation" wording, rename Params to Trainable Params or report both trainable/total, fix equation punctuation/formatting issues, and update limitation language.
4. Handle ChildPlay honestly: README says ChildPlay-gaze is not downloaded because YouTube is inaccessible. Do not fabricate results. Position GOO-real as the added OOD benchmark and leave ChildPlay as data-access-dependent future evaluation unless the user later obtains it.

Proposal requirements:
- Do not implement yet. Produce OpenSpec artifacts only.
- Include exact tasks, acceptance criteria, expected changed files, and how P0/P1 results should be inserted once available.
- Include a rebuttal skeleton with placeholders, but no fabricated experimental numbers.
- Keep scope focused on writing/reproducibility. Do not implement P0/P1 experiment scripts.
```

## Rebuttal Assembly Template

Use this structure once P0/P1/P2 results are available:

1. Opening
   - Thank reviewers.
   - State that we address the main concerns with new experiments and wording clarifications.

2. Clarifying GGSF
   - Acknowledge that "physical frustum" can be interpreted too strongly.
   - Clarify that GGSF is a frustum-inspired, head-conditioned spatial prior.
   - Explain how it is learned: relative coordinate tensor `(x-cx, y-cy, w, h)` -> lightweight MLP -> sigmoid mask -> multiplicative feature gating.

3. OOD Generalization
   - Insert GOO-real result table.
   - State whether GazeSpot improves over baseline and how this supports generalization.

4. Robustness to Noisy Head Boxes
   - Insert perturbation table/curve.
   - Discuss graceful degradation or failure modes honestly.

5. Efficiency
   - Insert trainable params, total params, FLOPs/MACs, latency, FPS.
   - Clarify that the original table reported trainable parameters.

6. Ablations and Feature Evidence
   - Insert simple mask/fusion ablations.
   - Insert quantitative over-smoothing analysis.

7. Crowd Subset Reproducibility
   - State construction rule, counts, release plan, and script/sample-ID availability.

8. Closing
   - Reiterate that the added evidence strengthens the central conclusion: multi-layer feature routing plus explicit head-conditioned spatial focus improves crowded gaze target estimation.
