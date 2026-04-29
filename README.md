# FahMai Directory Q&A — Competition Files

## Quick start

1. Build your RAG / tool-use system that reads `employees.csv` and answers `questions.csv`
2. Output a `submission.csv` with columns `id,response`
3. Grade locally: `python grade.py submission.csv train_labels.json`
4. Submit `submission.csv` to Kaggle

## Files

| File | Description |
|---|---|
| `data/employees.csv` | Employee directory (1,995 rows, 19 columns). Source of truth. |
| `data/questions.csv` | 300 questions (`id`, `language`, `question`). |
| `sample_submission.csv` | Submission template — 300 rows of `id,response`. |
| `train_labels.json` | Ground truth for 158 public-split items. Use for local grading. |
| `grade.py` | Local grader. Scores your submission against `train_labels.json`. |

## Local grading

```bash
python grade.py my_submission.csv train_labels.json
```

Output:
```
Scored 158 items against train_labels.json
Passed: 94/158 = 59.5%

Bucket                             pass/total    rate
--------------------------------------------------------
nickname_grid                    12/      17   70.6%
refuse                           11/      15   73.3%
...
```

The per-bucket breakdown shows which question types your system handles well vs. poorly.

## Sensitive data policy

Raw directory data and labels are local-only and must not be pushed to Git.
The repository ignore list blocks `employees.csv`, `questions.csv`,
`train_labels.json`, `FahMai Directory Q&A.pdf`, and `data/`.

Before pushing, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\safe-push.ps1 <remote> main
```

The safe-push script scans tracked files and blocks protected data paths or
likely secrets before it runs `git push`.

## Submission format

```csv
id,response
g001,The RETVP is Wiriya Chanchai (วิริยะ จันทชัย)
g002,OPSVP คือ คึกฤทธิ์ บุษราคัมวงศ์
...
```

- 300 rows, UTF-8 encoding
- Thai questions require Thai answers; English questions require English answers
- See `description.md` for refusal phrases and detailed evaluation rules
