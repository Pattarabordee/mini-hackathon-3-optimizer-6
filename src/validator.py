from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .jsonl_utils import read_jsonl
from .load_data import read_employees, read_questions
from .refusal_guard import (
    CANONICAL_REFUSALS,
    EMPLOYEE_ID_PATTERN,
    MOBILE_PATTERN,
    PHONE_EXTENSION_PATTERN,
    is_canonical_refusal,
    should_refuse,
)


THAI_CHAR_PATTERN = re.compile(r"[\u0E00-\u0E7F]")
LATIN_CHAR_PATTERN = re.compile(r"[A-Za-z]")


def validate_submission(root: Path) -> dict[str, Any]:
    submission_path = root / "outputs" / "submission.csv"
    report_path = root / "outputs" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    questions = read_questions(root)
    q_by_id = {row["id"]: row for row in questions}
    question_ids = [row["id"] for row in questions]

    if not submission_path.exists():
        errors.append({"type": "missing_submission", "message": str(submission_path)})
        report = _report(errors, warnings, summary)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    try:
        with submission_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except UnicodeDecodeError as exc:
        errors.append({"type": "utf8_error", "message": str(exc)})
        rows = []
        fieldnames = []

    if fieldnames != ["id", "response"]:
        errors.append({"type": "wrong_columns", "expected": ["id", "response"], "actual": fieldnames})

    ids = [row.get("id", "") for row in rows]
    summary.update(
        {
            "rows": len(rows),
            "expected_rows": len(question_ids),
            "unique_ids": len(set(ids)),
            "empty_responses": sum(not (row.get("response") or "").strip() for row in rows),
        }
    )

    if len(rows) != len(question_ids):
        errors.append({"type": "wrong_row_count", "expected": len(question_ids), "actual": len(rows)})

    duplicate_ids = sorted({iid for iid in ids if ids.count(iid) > 1})
    if duplicate_ids:
        errors.append({"type": "duplicate_id", "ids": duplicate_ids[:20], "count": len(duplicate_ids)})

    missing_ids = sorted(set(question_ids) - set(ids))
    extra_ids = sorted(set(ids) - set(question_ids))
    if missing_ids:
        errors.append({"type": "missing_id", "ids": missing_ids[:20], "count": len(missing_ids)})
    if extra_ids:
        errors.append({"type": "extra_id", "ids": extra_ids[:20], "count": len(extra_ids)})

    for row in rows:
        if not (row.get("response") or "").strip():
            errors.append({"type": "empty_response", "id": row.get("id")})

    analyses = read_jsonl(root / "cache" / "question_analysis.jsonl", key="id")
    evidence = read_jsonl(root / "outputs" / "evidence.jsonl", key="id")
    assert isinstance(analyses, dict)
    assert isinstance(evidence, dict)

    df = read_employees(root)
    mobile_values = {
        str(value).strip()
        for value in df.get("Mobile No.", [])
        if str(value).strip()
    }

    for row in rows:
        qid = row.get("id", "")
        response = (row.get("response") or "").strip()
        question = q_by_id.get(qid, {})
        language = question.get("language", "")
        analysis = analyses.get(qid, {})
        ev = evidence.get(qid, {})

        _validate_language(qid, language, response, warnings)
        _validate_leakage(qid, response, warnings)

        do_refuse, refusal_type = should_refuse(analysis, ev)
        if do_refuse:
            if not is_canonical_refusal(response, language):
                errors.append(
                    {
                        "type": "refusal_phrase_mismatch",
                        "id": qid,
                        "refusal_type": refusal_type,
                        "response": response,
                    }
                )
            if EMPLOYEE_ID_PATTERN.search(response) or PHONE_EXTENSION_PATTERN.search(response) or MOBILE_PATTERN.search(response):
                errors.append({"type": "refusal_leakage", "id": qid})

        for mobile in mobile_values:
            if mobile and mobile in response and not _evidence_contains_value(ev, mobile):
                warnings.append({"type": "mobile_value_without_evidence", "id": qid})
                break

        _validate_evidence_consistency(qid, response, ev, warnings)

    report = _report(errors, warnings, summary)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _validate_language(qid: str, language: str, response: str, warnings: list[dict[str, Any]]) -> None:
    if not response:
        return
    has_thai = bool(THAI_CHAR_PATTERN.search(response))
    has_latin = bool(LATIN_CHAR_PATTERN.search(response))
    if language == "th" and has_latin and not has_thai:
        warnings.append({"type": "wrong_language_suspected", "id": qid, "language": language})
    if language == "en" and has_thai and not has_latin:
        warnings.append({"type": "wrong_language_suspected", "id": qid, "language": language})


def _validate_leakage(qid: str, response: str, warnings: list[dict[str, Any]]) -> None:
    if EMPLOYEE_ID_PATTERN.search(response):
        warnings.append({"type": "employee_id_pattern", "id": qid})
    if PHONE_EXTENSION_PATTERN.search(response):
        warnings.append({"type": "phone_extension_pattern", "id": qid})
    if MOBILE_PATTERN.search(response):
        warnings.append({"type": "mobile_pattern", "id": qid})


def _validate_evidence_consistency(qid: str, response: str, evidence: dict[str, Any], warnings: list[dict[str, Any]]) -> None:
    rows = evidence.get("matched_rows") or []
    if evidence.get("retrieval_status") in {"found", "ambiguous"} and rows and response:
        row_text = json.dumps(rows[:20], ensure_ascii=False).casefold()
        response_tokens = [token.casefold() for token in re.findall(r"[A-Za-z\u0E00-\u0E7F]{3,}", response)]
        if response_tokens and not any(token in row_text for token in response_tokens):
            warnings.append({"type": "response_not_supported_by_evidence", "id": qid})


def _evidence_contains_value(evidence: dict[str, Any], value: str) -> bool:
    return value in json.dumps(evidence.get("matched_rows") or [], ensure_ascii=False)


def _report(errors: list[dict[str, Any]], warnings: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    summary = dict(summary)
    summary["error_count"] = len(errors)
    summary["warning_count"] = len(warnings)
    return {
        "status": "fail" if errors else "pass",
        "summary": summary,
        "errors": errors,
        "warnings": warnings,
        "canonical_refusals": CANONICAL_REFUSALS,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(validate_submission(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

