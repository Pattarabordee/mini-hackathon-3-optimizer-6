from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig
from .jsonl_utils import read_jsonl, write_jsonl
from .load_data import read_questions
from .refusal_guard import REFUSAL_TYPES
from .typhoon_client import TyphoonClient


INTENTS = {
    "lookup_person",
    "lookup_position",
    "lookup_nickname",
    "lookup_contact",
    "lookup_department",
    "lookup_section",
    "lookup_unit",
    "lookup_office",
    "lookup_branch",
    "count_employees",
    "list_employees",
    "compare_or_filter",
    "field_not_in_directory",
    "person_not_found",
    "speculation_or_opinion",
    "external_company",
    "prompt_injection",
    "field_exists_but_blank",
    "unknown",
}

ENTITY_KEYS = [
    "person_name",
    "nickname",
    "position",
    "department",
    "section",
    "unit",
    "office",
    "branch",
    "field",
    "other",
]


def analysis_cache_path(config: AppConfig) -> Path:
    return config.cache_dir / "question_analysis.jsonl"


def analyze_questions(config: AppConfig, force: bool = False) -> tuple[list[dict[str, Any]], int]:
    questions = read_questions(config.root)
    cache_path = analysis_cache_path(config)
    cached = {} if force else read_jsonl(cache_path, key="id")
    assert isinstance(cached, dict)

    client = TyphoonClient(config)
    records: list[dict[str, Any]] = []
    api_calls_before = client.real_api_calls

    for question in questions:
        qid = question["id"]
        if qid in cached:
            records.append(cached[qid])
            continue
        records.append(_analyze_one(client, question, config.max_retries))

    write_jsonl(cache_path, records)
    return records, client.real_api_calls - api_calls_before


def _analyze_one(client: TyphoonClient, question: dict[str, str], max_json_retries: int) -> dict[str, Any]:
    last_error = ""
    for attempt in range(max_json_retries + 1):
        result = client.chat(_analysis_messages(question, attempt), temperature=0.0)
        if not result.ok:
            last_error = result.error or "Typhoon request failed"
            continue
        try:
            parsed = parse_json_object(result.content)
            return normalize_analysis(question, parsed, error=None)
        except Exception as exc:  # noqa: BLE001 - stored in analysis notes
            last_error = str(exc)
    return normalize_analysis(question, {}, error=f"invalid_json: {last_error}")


def _analysis_messages(question: dict[str, str], attempt: int) -> list[dict[str, str]]:
    taxonomy = ", ".join(sorted(INTENTS))
    refusal_types = ", ".join("null" if item is None else item for item in REFUSAL_TYPES)
    repair = " Previous output was invalid JSON. Return only a valid JSON object." if attempt else ""
    payload = {
        "id": question.get("id", ""),
        "language": question.get("language", ""),
        "question": question.get("question", ""),
    }
    system = (
        "You are the only allowed LLM question analyzer for the competition. "
        "Analyze the directory question and return JSON only. "
        "Do not answer the question. Do not include prose. "
        "QUESTION_ANALYSIS_SCHEMA. "
        f"Allowed intents: {taxonomy}. Allowed refusal_type values: {refusal_types}."
    )
    user = (
        "Return exactly one JSON object with keys: id, language, question, intent, entities, "
        "target_fields, needs_refusal, refusal_type, requires_count, requires_list, "
        "requires_exact_count, notes. Entities must contain person_name, nickname, position, "
        "department, section, unit, office, branch, field, other. "
        f"{repair}\nQuestion JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        value, _ = decoder.raw_decode(cleaned[index:])
        if isinstance(value, dict):
            return value
    raise ValueError("No JSON object found")


def normalize_analysis(question: dict[str, str], raw: dict[str, Any], error: str | None) -> dict[str, Any]:
    intent = raw.get("intent") if raw.get("intent") in INTENTS else "unknown"
    refusal_type = raw.get("refusal_type")
    if refusal_type not in REFUSAL_TYPES:
        refusal_type = None

    entities = raw.get("entities") if isinstance(raw.get("entities"), dict) else {}
    normalized_entities: dict[str, Any] = {}
    for key in ENTITY_KEYS:
        value = entities.get(key)
        if key == "other":
            normalized_entities[key] = value if isinstance(value, list) else []
        else:
            normalized_entities[key] = None if value is None or value == "" or value == [] else value

    return {
        "id": question.get("id", raw.get("id", "")),
        "language": question.get("language", raw.get("language", "en")),
        "question": question.get("question", raw.get("question", "")),
        "intent": intent,
        "entities": normalized_entities,
        "target_fields": raw.get("target_fields") if isinstance(raw.get("target_fields"), list) else [],
        "needs_refusal": bool(raw.get("needs_refusal", False)),
        "refusal_type": refusal_type,
        "requires_count": bool(raw.get("requires_count", False)),
        "requires_list": bool(raw.get("requires_list", False)),
        "requires_exact_count": bool(raw.get("requires_exact_count", False)),
        "notes": (raw.get("notes") or "") + (f" | {error}" if error else ""),
        "analysis_error": error,
    }
