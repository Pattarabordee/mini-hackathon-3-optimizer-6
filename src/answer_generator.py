from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import AppConfig
from .jsonl_utils import read_jsonl, write_jsonl
from .load_data import read_questions
from .question_router import parse_json_object
from .refusal_guard import apply_refusal
from .typhoon_client import TyphoonClient


IDENTITY_COLUMNS = [
    "First Name Thai",
    "Last Name Thai",
    "First Name English",
    "Last Name English",
]


def generated_cache_path(config: AppConfig) -> Path:
    return config.cache_dir / "generated_answers.jsonl"


def generate_answers(
    config: AppConfig,
    analyses: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]],
    force: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    cached = {} if force else read_jsonl(generated_cache_path(config), key="id")
    assert isinstance(cached, dict)

    evidence_by_id = {record["id"]: record for record in evidence_records}
    client = TyphoonClient(config)
    records: list[dict[str, Any]] = []
    api_calls_before = client.real_api_calls

    for analysis in analyses:
        qid = analysis["id"]
        if qid in cached:
            records.append(cached[qid])
            continue
        evidence = evidence_by_id.get(qid, {})
        answer_record = generate_one(config, client, analysis, evidence)
        records.append(answer_record)

    write_jsonl(generated_cache_path(config), records)
    write_submission(config.root, records)
    return records, client.real_api_calls - api_calls_before


def generate_one(config: AppConfig, client: TyphoonClient, analysis: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    refusal = apply_refusal(analysis, evidence)
    if refusal is not None:
        return {
            "id": analysis["id"],
            "response": refusal,
            "source": "refusal_guard",
            "mock": False,
            "generation_notes": "Canonical refusal override.",
        }

    deterministic = deterministic_answer(analysis, evidence)
    if deterministic:
        return {
            "id": analysis["id"],
            "response": deterministic,
            "source": "deterministic",
            "mock": False,
            "generation_notes": "Formatted directly from evidence.",
        }

    no_evidence = evidence.get("retrieval_status") in {None, "not_found", "error"}
    if no_evidence:
        return {
            "id": analysis["id"],
            "response": fallback_answer(config, analysis),
            "source": "no_evidence_fallback",
            "mock": config.use_mock_typhoon,
            "generation_notes": "Skipped Typhoon answer generation because retrieval returned no usable evidence.",
        }

    result = client.chat(_answer_messages(analysis, evidence), temperature=0.0)
    if result.ok:
        try:
            parsed = parse_json_object(result.content)
            response = str(parsed.get("response", "")).strip()
        except Exception:  # noqa: BLE001 - fallback to raw content
            response = result.content.strip()
        if response:
            return {
                "id": analysis["id"],
                "response": response,
                "source": "typhoon_mock" if result.mock else "typhoon",
                "mock": result.mock,
                "generation_notes": "Generated from limited evidence by Typhoon-compatible client.",
            }

    return {
        "id": analysis["id"],
        "response": fallback_answer(config, analysis),
        "source": "fallback",
        "mock": config.use_mock_typhoon,
        "generation_notes": result.error if not result.ok else "No non-empty Typhoon response.",
    }


def fallback_answer(config: AppConfig, analysis: dict[str, Any]) -> str:
    language = analysis.get("language", "en")
    if config.use_mock_typhoon:
        if language == "th":
            return "MOCK_TYPHOON: ไม่ใช่คำตอบจริงจาก Typhoon"
        return "MOCK_TYPHOON: not a real Typhoon answer"
    if language == "th":
        return "ไม่พบข้อมูล"
    return "no record found"


def deterministic_answer(analysis: dict[str, Any], evidence: dict[str, Any]) -> str | None:
    status = evidence.get("retrieval_status")
    language = analysis.get("language", "en")

    if status not in {"found", "ambiguous"}:
        return None

    computed_value = evidence.get("computed_value")
    if computed_value is not None:
        return f"จำนวน {computed_value} คน" if language == "th" else str(computed_value)

    rows = evidence.get("matched_rows") or []
    columns = evidence.get("matched_columns") or []
    if not rows:
        return None

    if analysis.get("requires_list") or status == "ambiguous" or len(rows) > 1:
        names = [format_name(row, language) for row in rows if format_name(row, language)]
        if not names:
            return None
        return "; ".join(names)

    row = rows[0]
    if columns:
        if any(column in IDENTITY_COLUMNS for column in columns):
            return format_name(row, language) or None
        parts = []
        for column in language_preferred_columns(columns, language):
            value = str(row.get(column, "")).strip()
            if value:
                parts.append(value)
        if parts:
            return ", ".join(parts)

    return format_name(row, language) or None


def language_preferred_columns(columns: list[str], language: str) -> list[str]:
    preferred: list[str] = []
    for column in columns:
        if language == "th":
            column = {
                "Nickname English": "Nickname Thai",
                "Position in English": "Position in Thai",
            }.get(column, column)
        elif language == "en":
            column = {
                "Nickname Thai": "Nickname English",
                "Position in Thai": "Position in English",
            }.get(column, column)
        preferred.append(column)
    preferred.extend(columns)
    return list(dict.fromkeys(preferred))


def format_name(row: dict[str, Any], language: str) -> str:
    thai = " ".join(str(row.get(column, "")).strip() for column in ["First Name Thai", "Last Name Thai"]).strip()
    english = " ".join(str(row.get(column, "")).strip() for column in ["First Name English", "Last Name English"]).strip()
    if language == "th":
        if thai and english:
            return f"{thai} ({english})"
        return thai or english
    if english and thai:
        return f"{english} ({thai})"
    return english or thai


def _answer_messages(analysis: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, str]]:
    limited_evidence = {
        "retrieval_status": evidence.get("retrieval_status"),
        "matched_columns": evidence.get("matched_columns", []),
        "computed_value": evidence.get("computed_value"),
        "matched_rows": (evidence.get("matched_rows") or [])[:20],
        "retrieval_notes": evidence.get("retrieval_notes", ""),
    }
    payload = {
        "id": analysis.get("id"),
        "language": analysis.get("language"),
        "question": analysis.get("question"),
        "intent": analysis.get("intent"),
        "target_fields": analysis.get("target_fields", []),
        "evidence": limited_evidence,
    }
    system = (
        "You are the only allowed answer generator LLM for this competition. "
        "Use only the provided evidence. Do not guess outside evidence. "
        "Keep the answer short and in the requested language. "
        "Return JSON only as {\"response\": \"...\"}. ANSWER_GENERATION_SCHEMA."
    )
    user = json.dumps(payload, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def write_submission(root: Path, answer_records: list[dict[str, Any]]) -> Path:
    answers_by_id = {record["id"]: record["response"] for record in answer_records}
    questions = read_questions(root)
    output_path = root / "outputs" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        for question in questions:
            writer.writerow({"id": question["id"], "response": answers_by_id.get(question["id"], "")})
    return output_path
