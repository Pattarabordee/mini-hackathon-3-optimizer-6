from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_INPUT_FILES = [
    "employees.csv",
    "questions.csv",
    "sample_submission.csv",
    "train_labels.json",
    "grade.py",
]


def missing_required_files(root: Path) -> list[str]:
    return [name for name in REQUIRED_INPUT_FILES if not (root / name).exists()]


def write_missing_files_report(root: Path, missing: list[str]) -> Path:
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "missing_files.md"
    lines = [
        "# Missing Required Files",
        "",
        "The local harness cannot run because required input files are missing.",
        "",
        "## Missing Files",
        "",
        *[f"- `{name}`" for name in missing],
        "",
        "## Impact",
        "",
        "- The pipeline will not infer schemas or create dummy competition inputs.",
        "- Restore the missing files, then rerun the harness.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def ensure_required_files(root: Path) -> None:
    missing = missing_required_files(root)
    if missing:
        report = write_missing_files_report(root, missing)
        raise FileNotFoundError(f"Missing required files: {missing}. See {report}")


def read_employees(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "employees.csv", dtype=str, keep_default_na=False)


def read_questions(root: Path) -> list[dict[str, str]]:
    with (root / "questions.csv").open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_sample_submission(root: Path) -> list[dict[str, str]]:
    with (root / "sample_submission.csv").open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_train_labels(root: Path) -> dict[str, Any]:
    return json.loads((root / "train_labels.json").read_text(encoding="utf-8-sig"))

