# FahMai Directory Q&A Harness

Typhoon v2.5 local harness for FahMai Directory Q&A Minihack with deterministic retrieval, validation, caching, and pre-submit grading.

This repository is the Team Optimizer_6 workspace for Mini Hackathon 3. It builds a local harness around the required Typhoon model, validates submissions before Kaggle upload, and keeps sensitive directory data out of Git.

## Competition Constraints

- LLM-based question analysis and answer generation must use only `typhoon-v2.5-30b-a3b-instruct`.
- Do not use GPT, Claude, Gemini, NotebookLM, Codex reasoning, or any other LLM to analyze real questions or generate final answers.
- Codex may write and maintain the harness code only.
- Do not submit to Kaggle automatically.
- Cache every Typhoon call.
- Keep refusal phrases exact.

## Required Local Files

These files must exist in the project root:

- `employees.csv`
- `questions.csv`
- `sample_submission.csv`
- `train_labels.json`
- `grade.py`

The raw CSV/JSON/PDF data files are ignored by Git and must be shared through an approved secure channel.

## Setup

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`:

```text
TYPHOON_API_KEY=
TYPHOON_BASE_URL=https://api.opentyphoon.ai/v1
TYPHOON_MODEL=typhoon-v2.5-30b-a3b-instruct
MOCK_TYPHOON=false
TYPHOON_TIMEOUT_SECONDS=60
TYPHOON_MAX_RETRIES=2
```

If `TYPHOON_API_KEY` is empty or `MOCK_TYPHOON=true`, the harness runs in mock mode. Mock mode is for testing the software pipeline only and does not create real competition answers.

## Run Full Pipeline

```powershell
python -m src.run_pipeline
```

Workflow:

1. Check required files
2. Profile `employees.csv`
3. Analyze questions with Typhoon and cache to `cache/question_analysis.jsonl`
4. Retrieve deterministic evidence to `outputs/evidence.jsonl`
5. Generate answers to `cache/generated_answers.jsonl`
6. Write `outputs/submission.csv`
7. Validate submission
8. Run local grading with `grade.py`
9. Write reports

## Validation Only

```powershell
python -m src.run_pipeline --validate-only
```

Output:

- `outputs/validation_report.json`

## Local Grading Only

```powershell
python -m src.run_pipeline --grade-only
```

Equivalent grading command:

```powershell
python grade.py outputs/submission.csv train_labels.json
```

## Reports

- `reports/data_profile.md`: technical profile with sensitive sample values redacted
- `reports/run_report.md`: pipeline run summary, validator status, grading output
- `reports/error_analysis.md`: public-label failure categories and recommended fixes
- `reports/missing_files.md`: generated only when required inputs are missing

## Tests

```powershell
pytest
```

## Before Kaggle Submission

Submit only after:

- `outputs/submission.csv` exists with exactly 300 rows and columns `id,response`
- Validator status is `pass` or has no critical errors
- Local public score is high enough for the team threshold
- Refusal phrase checks pass 100%
- There are no empty responses
- There are no critical wrong-language issues
- There is no leakage in refusal responses
- A human reviewer has inspected `reports/error_analysis.md`

## Safety Rules

- Never commit `.env`, API keys, Kaggle tokens, raw credentials, or generated submissions that have not been reviewed.
- Raw data files such as `employees.csv`, `questions.csv`, `train_labels.json`, and `FahMai Directory Q&A.pdf` stay local.
- Use `scripts/safe-push.ps1` when pushing to GitHub:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\safe-push.ps1 origin main:main
```

## Known Limitations

- Mock mode validates the harness only; it cannot produce competition-grade answers.
- Deterministic retrieval depends on Typhoon question analysis quality.
- Public labels cover only part of the competition set, so avoid overfitting fixes to public failed ids.

