from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .jsonl_utils import read_jsonl


def load_submission(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["id"].strip(): (row.get("response") or "").strip() for row in csv.DictReader(f)}


def grade_item(gt: dict[str, Any], response: str) -> tuple[bool, list[str]]:
    expected_answer = gt.get("expected_answer") or {}
    response_lower = response.lower()
    failures: list[str] = []

    for group in expected_answer.get("must_contain_any_of", []):
        if group and not any(token and str(token).lower() in response_lower for token in group):
            failures.append("missing_keyword")

    for bad in expected_answer.get("must_not_contain", []):
        if bad and str(bad).lower() in response_lower:
            failures.append("forbidden_keyword")

    if expected_answer.get("must_not_contain_phone_extension") and re.search(r"\b\d{5}\b", response):
        failures.append("refusal_leaked_phone_extension")
    if expected_answer.get("must_not_contain_employee_id_pattern") and re.search(r"\b0[08]\d{6}\b", response):
        failures.append("refusal_leaked_employee_id")

    tokens_per_id = expected_answer.get("all_items_tokens_per_id") or {}
    if tokens_per_id:
        matched_ids = []
        for employee_id, tokens in tokens_per_id.items():
            if tokens and any(token and str(token).lower() in response_lower for token in tokens):
                matched_ids.append(employee_id)
        min_items = expected_answer.get("min_items")
        if min_items is not None and len(matched_ids) < min_items:
            failures.append("missing_employee")
        exact_count = expected_answer.get("exact_count")
        if exact_count is not None and len(matched_ids) != exact_count:
            failures.append("wrong_exact_count")

    return not failures, failures


def build_error_analysis(root: Path, grade_stdout: str = "", grade_stderr: str = "") -> Path:
    submission_path = root / "outputs" / "submission.csv"
    report_path = root / "reports" / "error_analysis.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    questions = {}
    with (root / "questions.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            questions[row["id"]] = row

    labels = json.loads((root / "train_labels.json").read_text(encoding="utf-8-sig"))["items"]
    submissions = load_submission(submission_path) if submission_path.exists() else {}
    evidence = read_jsonl(root / "outputs" / "evidence.jsonl", key="id")
    assert isinstance(evidence, dict)
    validation_report = {}
    validation_path = root / "outputs" / "validation_report.json"
    if validation_path.exists():
        validation_report = json.loads(validation_path.read_text(encoding="utf-8"))

    validation_warnings_by_id: dict[str, list[str]] = defaultdict(list)
    for warning in validation_report.get("warnings", []):
        if warning.get("id"):
            validation_warnings_by_id[warning["id"]].append(warning.get("type", "validation_warning"))

    failures: list[dict[str, Any]] = []
    passed = 0
    for gt in labels:
        qid = gt["id"]
        response = submissions.get(qid, "")
        ok, reasons = grade_item(gt, response)
        if ok:
            passed += 1
            continue
        ev = evidence.get(qid, {})
        category = suspect_failure_category(reasons, ev, validation_warnings_by_id.get(qid, []), response)
        failures.append(
            {
                "id": qid,
                "language": questions.get(qid, {}).get("language", gt.get("language", "")),
                "bucket": gt.get("bucket"),
                "response": response,
                "evidence_status": ev.get("retrieval_status"),
                "evidence_summary": summarize_evidence(ev),
                "expected_constraints": summarize_expected(gt.get("expected_answer") or {}),
                "reasons": reasons,
                "suspected_failure_reason": category,
                "recommended_fix": recommended_fix(category),
            }
        )

    category_counts = Counter(item["suspected_failure_reason"] for item in failures)
    bucket_counts = Counter(item["bucket"] for item in failures)

    lines = [
        "# Error Analysis",
        "",
        "This report is generated from public labels only. It must not be used to hard-code answers by id.",
        "",
        "## Local Score Summary",
        "",
        f"- Public labeled items: {len(labels)}",
        f"- Passed: {passed}",
        f"- Failed: {len(failures)}",
        f"- Score: {passed / len(labels):.4f}" if labels else "- Score: n/a",
        "",
        "## Grade Command Output",
        "",
        "```text",
        (grade_stdout or "").strip() or "(no stdout captured)",
        "```",
        "",
    ]
    if grade_stderr.strip():
        lines.extend(["## Grade Command Errors", "", "```text", grade_stderr.strip(), "```", ""])

    lines.extend(["## Top 10 Failure Categories", ""])
    for category, count in category_counts.most_common(10):
        lines.append(f"- {category}: {count}")
    if not category_counts:
        lines.append("- none")

    lines.extend(["", "## Top Failed Buckets", ""])
    for bucket, count in bucket_counts.most_common(10):
        lines.append(f"- {bucket}: {count}")
    if not bucket_counts:
        lines.append("- none")

    lines.extend(["", "## Failed Items", ""])
    for item in failures:
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- Language: {item['language']}",
                f"- Bucket: {item['bucket']}",
                f"- Generated response: `{item['response']}`",
                f"- Evidence: {item['evidence_summary']}",
                f"- Expected constraints: {item['expected_constraints']}",
                f"- Suspected failure reason: {item['suspected_failure_reason']}",
                f"- Recommended fix: {item['recommended_fix']}",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def suspect_failure_category(
    reasons: list[str],
    evidence: dict[str, Any],
    validation_warnings: list[str],
    response: str,
) -> str:
    if "wrong_language_suspected" in validation_warnings:
        return "wrong_language"
    if "refusal_leaked_phone_extension" in reasons or "refusal_leaked_employee_id" in reasons:
        return "refusal_phrase_mismatch"
    if "wrong_exact_count" in reasons:
        return "wrong_exact_count"
    if "missing_employee" in reasons:
        return "missing_employee"
    if evidence.get("retrieval_status") in {None, "not_found", "error"}:
        return "retrieval_failure"
    if "response_not_supported_by_evidence" in validation_warnings:
        return "hallucination"
    if not response.strip():
        return "formatting_issue"
    if "missing_keyword" in reasons:
        return "missing_keyword"
    if "forbidden_keyword" in reasons:
        return "extra_employee"
    return "unknown"


def summarize_evidence(evidence: dict[str, Any]) -> str:
    rows = evidence.get("matched_rows") or []
    return (
        f"status={evidence.get('retrieval_status')}, rows={len(rows)}, "
        f"columns={evidence.get('matched_columns')}, computed={evidence.get('computed_value')}"
    )


def summarize_expected(expected: dict[str, Any]) -> str:
    keys = sorted(expected.keys())
    parts = [f"keys={keys}"]
    if expected.get("min_items") is not None:
        parts.append(f"min_items={expected.get('min_items')}")
    if expected.get("exact_count") is not None:
        parts.append(f"exact_count={expected.get('exact_count')}")
    if expected.get("must_not_contain_phone_extension"):
        parts.append("no_phone_extension")
    if expected.get("must_not_contain_employee_id_pattern"):
        parts.append("no_employee_id")
    return ", ".join(parts)


def recommended_fix(category: str) -> str:
    fixes = {
        "wrong_language": "Tighten language-specific formatting in answer_generator.",
        "missing_keyword": "Improve retrieval target fields or deterministic answer formatting for this bucket.",
        "wrong_exact_count": "Inspect count filter logic and entity normalization.",
        "missing_employee": "Improve list retrieval recall and candidate formatting.",
        "extra_employee": "Reduce over-broad filters before list formatting.",
        "refusal_phrase_mismatch": "Route this case through refusal_guard canonical phrases.",
        "retrieval_failure": "Improve Typhoon analysis schema quality or deterministic entity matching.",
        "hallucination": "Restrict answer generation to matched evidence fields.",
        "formatting_issue": "Ensure submission response is non-empty and concise.",
        "ambiguous_question": "Add deterministic ambiguity handling with evidence lists.",
    }
    return fixes.get(category, "Human review required before prompt or retrieval iteration.")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(build_error_analysis(root))


if __name__ == "__main__":
    main()

