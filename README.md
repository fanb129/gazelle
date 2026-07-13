# Gazelle AAAI2027 Workspace

This repository is now configured as the working branch for the AAAI 2027 submission cycle.

## Current Direction

- Active branch: `codex/aaai2027`.
- All next research, engineering, experiments, writing, and review work in this repository should target AAAI 2027.
- The previous V1 / ACM MM 2026 rebuttal work is no longer the active target. Treat the ACM MM 2026 material as historical context only.
- Local changes from the previous `v1` workspace were preserved in Git stash entries before this branch was created:
  - `stash@{0}: On v1: pre-aaai2027-untracked-rebuttal-files`
  - `stash@{1}: On v1: pre-aaai2027-cleanup-from-v1`

## Codex Skills

Supervisor-Skills has been installed as project-local Skills only.

- Skill root: `.agents/skills/`
- Source repository: `https://github.com/HKUSTDial/Supervisor-Skills`
- Do not install or depend on copies under `$HOME/.agents/skills`, `~/.codex/skills`, or any other global/user/system directory for this project.
- Each skill is kept in its own directory with its original `SKILL.md` and related resources.

Installed skill directories:

- `.agents/skills/benchmark-paper-template/`
- `.agents/skills/deep-research/`
- `.agents/skills/drawio-reconstruction/`
- `.agents/skills/figure-designer/`
- `.agents/skills/idea-evaluator/`
- `.agents/skills/intro-drafter/`
- `.agents/skills/paper-polish/`
- `.agents/skills/paper-writer/`
- `.agents/skills/pre-submission-reviewer/`
- `.agents/skills/tech-paper-template/`
- `.agents/skills/vibe-research-workflow/`

## Development And Execution Setup

- Local environment: used for code edits, repository cleanup, test edits, and documentation updates.
- Real training/evaluation runs: performed on the server after the user pulls the updated code and manually starts jobs.
- Server login: `fb@3090.lab`; local SSH key login is already configured.
- Main Python package: `gazelle`, defined by `setup.py`.
- Existing command notes and experiment snippets are in `help.md`.

Common script defaults and paths currently assume the server dataset layout:

- GazeFollow: `/newhome/fb/dataset/gazefollow_extended`
- VideoAttentionTarget: `/newhome/fb/dataset/videoattentiontarget`
- GooReal: `/newhome/fb/dataset/gooreal_data`
- Default checkpoint/output root in scripts: `./experiments`
- Default image output root in scripts: `./experiments_imgs`
- Existing scripts still use `gazelleV1` as the default Weights & Biases project name; review this before launching new AAAI 2027 experiments.

## Dataset Notes

GooReal has already been downloaded on the server:

```text
fb@nise-server:/newhome/fb/dataset/gooreal_data
├── gooreal.zip
├── oneshotrealhumansNew.pickle
├── testrealhumansNew.pickle
├── testrealhumansSparsedNew.pickle
├── valrealhumansNew.pickle
├── wget-log
├── wget-log.1
├── wget-log.2
├── wget-log.3
└── wget-log.4
```

ChildPlay-gaze has not been downloaded yet because YouTube access was unavailable.

## Working Rule

When planning or modifying this repository from now on, assume the objective is AAAI 2027 unless explicitly told otherwise.
