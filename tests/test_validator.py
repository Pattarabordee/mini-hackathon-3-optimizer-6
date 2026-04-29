import csv
import json
from pathlib import Path

import pytest

from src.config import ConfigError, load_config
from src.validator import validate_submission


def _write_minimal_project(root: Path):
    (root / "cache").mkdir()
    (root / "outputs").mkdir()
    (root / "employees.csv").write_text(
        "Employee ID,Department,Section,Unit,Position in Thai,Position in English,First Name Thai,Last Name Thai,First Name English,Last Name English,Nickname Thai,Nickname English,Email Address,Phone Extension,Mobile No.,Office Location,Branch,Start Year,Position Level\n"
        "0800001,TEC,TEC-INF,TEC-INF-1,วิศวกร,Engineer,สมชาย,ใจดี,Somchai,Jaidee,ชัย,Chai,somchai@example.com,12345,0812345678,HQ,BKK,2024,IC\n",
        encoding="utf-8",
    )
    (root / "questions.csv").write_text("id,language,question\nq1,en,Who is test\n", encoding="utf-8")
    (root / "cache" / "question_analysis.jsonl").write_text(
        json.dumps({"id": "q1", "language": "en", "needs_refusal": False}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "outputs" / "evidence.jsonl").write_text(
        json.dumps({"id": "q1", "retrieval_status": "not_found"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_submission(root: Path, rows, fieldnames=("id", "response")):
    with (root / "outputs" / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_validator_catches_empty_response(tmp_path):
    _write_minimal_project(tmp_path)
    _write_submission(tmp_path, [{"id": "q1", "response": ""}])
    report = validate_submission(tmp_path)
    assert report["status"] == "fail"
    assert any(error["type"] == "empty_response" for error in report["errors"])


def test_validator_catches_wrong_columns(tmp_path):
    _write_minimal_project(tmp_path)
    _write_submission(tmp_path, [{"id": "q1", "answer": "x"}], fieldnames=("id", "answer"))
    report = validate_submission(tmp_path)
    assert any(error["type"] == "wrong_columns" for error in report["errors"])


def test_validator_catches_duplicate_id(tmp_path):
    _write_minimal_project(tmp_path)
    _write_submission(tmp_path, [{"id": "q1", "response": "a"}, {"id": "q1", "response": "b"}])
    report = validate_submission(tmp_path)
    assert any(error["type"] == "duplicate_id" for error in report["errors"])


def test_config_fails_wrong_typhoon_model(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPHOON_MODEL", "typhoon-other")
    with pytest.raises(ConfigError):
        load_config(tmp_path)

