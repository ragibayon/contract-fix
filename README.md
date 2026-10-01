# ContractFix

ContractFix turns a repository issue into behavioral guidance, checks an executable contract when one can be qualified, and uses the resulting guidance to generate and select a repair patch. This release runs the **Python** workflow. The study task IDs for other languages are included as frozen cohort records.

## Requirements

- Python 3.12 or newer, [`uv`](https://docs.astral.sh/uv/), and Git.
- Docker with a running daemon and the task's image already available locally. ContractFix does not pull images during a run.
- An OpenRouter API key with access to the model specified in `.env.example`. Model calls may incur charges.

## Install and check

From the repository root:

```bash
uv sync --frozen
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. The other entries in `.env.example` are the supplied model settings. Keep `.env` private; it is ignored by Git. Check the installation without making a model call:

```bash
uv run contractfix doctor
```

## Run a Python issue

Prepare a JSON task with the issue and its repository identity:

```json
{
  "instance_id": "example__repo-123",
  "repo": "example/repo",
  "base_commit": "0123456789abcdef0123456789abcdef01234567",
  "version": "1.0",
  "problem_statement": "Describe the reported behavior and the expected behavior.",
  "image": "your-local-task-image:tag"
}
```

`--repo` must point to a checkout whose `HEAD` is `base_commit`, with no tracked changes. The Docker image named in the task must already exist locally. Put `--out` outside the checkout.

```bash
uv run contractfix --env .env run \
  --workflow configs/workflow.json \
  --task /path/to/task.json \
  --repo /path/to/checkout \
  --out results/issue-1
```

The supplied workflow qualifies contracts and attempts repair in the same run. Inspect `results/issue-1/summary.json` for the status and `results/issue-1/repair/selected.patch` when a patch is selected. The run reports its own patch checks; benchmark resolution requires the benchmark evaluator.

## Frozen study tasks

The `tasks/` directory contains the exact task IDs used in the study:

| Dataset | Development | Evaluation |
| --- | ---: | ---: |
| SWE-bench Lite, Python | 183, plus 23 validation | — |
| SWE-bench Verified, Python | — | 218 |
| SWE-bench Multilingual, Java/C/C++ | 26 | 59 |
| SWE-bench Multilingual, JavaScript ecosystem | 13 | 30 |

`tasks/cohorts.json` identifies the dataset, split, and pinned revision for every cohort. Validate the lists without downloading anything:

```bash
uv run python scripts/prepare_tasks.py
```

To load the pinned datasets and export the selected issue records, run:

```bash
uv run --with 'datasets==5.0.1' python scripts/prepare_tasks.py --download
```

The script checks the Verified and JavaScript samples against their source datasets and writes the selected records to `downloads/tasks/`. `sources.json` records the source revisions. The JSONL records contain `instance_id`, `repo`, `base_commit`, `version`, and `problem_statement`; multilingual records also include `image`. Reference patches, benchmark tests, hints, and evaluation outcomes are excluded. The downloaded directory is ignored by Git.

## Motivating example

`motivating_example/django__django-12965/` contains the task record, ContractFix, SpecRover, and ACR predictions, available reports, and the two paired ContractFix selection variants. These files are example evidence; run the task through the command above with a clean Django checkout at the commit in `task.json` and its named Docker image to produce a new ContractFix run.
