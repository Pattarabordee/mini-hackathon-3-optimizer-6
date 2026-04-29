# AGENTS.md

## Project Mission

Build a local evaluation harness for the FahMai Directory Q&A Minihack.

## Role Boundary

Codex is used only to write, test, and maintain the software harness.

Codex must not analyze real competition questions, infer real answers, or generate final competition responses using non-Typhoon reasoning.

All LLM-based question analysis and answer generation must be performed only by:

`typhoon-v2.5-30b-a3b-instruct`

## Hard Rules

- Use only `typhoon-v2.5-30b-a3b-instruct` for all LLM-based question analysis and answer generation.
- Do not use GPT, Claude, Gemini, NotebookLM, Codex reasoning, or any non-Typhoon LLM to analyze questions or generate answers.
- Do not use other Typhoon model variants unless the competition organizer explicitly updates the rule.
- Do not submit to Kaggle automatically.
- Prefer deterministic CSV filtering over LLM guessing.
- Cache every Typhoon call.
- Keep answers in the same language as the question.
- Keep refusal phrases exact.
- Never leak Employee ID or phone extension in refusal responses.
- Do not hard-code API keys or secrets.
- Make small, reviewable diffs.
- Run validator before claiming success.

## Done Means

- outputs/submission.csv exists.
- outputs/validation_report.json exists.
- reports/run_report.md exists.
- reports/error_analysis.md exists.
- Local grading has been attempted if grade.py and train_labels.json are available.
- Any failure or missing file is documented clearly.

