from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from .jsonl_utils import read_jsonl, write_jsonl
from .load_data import read_employees


FIELD_TO_COLUMNS = {
    "employee_id": ["Employee ID"],
    "id": ["Employee ID"],
    "name": ["First Name Thai", "Last Name Thai", "First Name English", "Last Name English"],
    "thai_name": ["First Name Thai", "Last Name Thai"],
    "english_name": ["First Name English", "Last Name English"],
    "nickname": ["Nickname Thai", "Nickname English"],
    "thai_nickname": ["Nickname Thai"],
    "english_nickname": ["Nickname English"],
    "position": ["Position in Thai", "Position in English"],
    "department": ["Department"],
    "section": ["Section"],
    "unit": ["Unit"],
    "office": ["Office Location"],
    "location": ["Office Location"],
    "branch": ["Branch"],
    "email": ["Email Address"],
    "phone": ["Phone Extension"],
    "extension": ["Phone Extension"],
    "mobile": ["Mobile No."],
    "start_year": ["Start Year"],
    "position_level": ["Position Level"],
}

ENTITY_FIELD_TO_COLUMN = {
    "department": "Department",
    "section": "Section",
    "unit": "Unit",
    "office": "Office Location",
    "branch": "Branch",
    "position": "Position in English",
}

REFUSAL_INTENTS = {
    "field_not_in_directory",
    "person_not_found",
    "speculation_or_opinion",
    "external_company",
    "prompt_injection",
    "field_exists_but_blank",
}

MAX_EVIDENCE_ROWS = 200


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = re.sub(r"[\s\-_.,/()]+", " ", text)
    return text.strip()


def retrieve_all(root: Path, analyses: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    df = read_employees(root)
    if analyses is None:
        analyses_obj = read_jsonl(root / "cache" / "question_analysis.jsonl")
        assert isinstance(analyses_obj, list)
        analyses = analyses_obj
    records = [retrieve_one(df, analysis) for analysis in analyses]
    write_jsonl(root / "outputs" / "evidence.jsonl", records)
    return records


def retrieve_one(df: pd.DataFrame, analysis: dict[str, Any]) -> dict[str, Any]:
    base = {
        "id": analysis.get("id", ""),
        "intent": analysis.get("intent", "unknown"),
        "matched_rows": [],
        "matched_columns": [],
        "computed_value": None,
        "confidence": 0.0,
        "retrieval_status": "not_found",
        "retrieval_notes": "",
    }

    intent = analysis.get("intent", "unknown")
    if intent in REFUSAL_INTENTS:
        base.update(
            {
                "retrieval_status": "refusal",
                "refusal_type": intent,
                "confidence": 1.0,
                "retrieval_notes": "Typhoon analysis classified this as a refusal intent.",
            }
        )
        return base

    if analysis.get("needs_refusal") and analysis.get("refusal_type"):
        base.update(
            {
                "retrieval_status": "refusal",
                "refusal_type": analysis.get("refusal_type"),
                "confidence": 1.0,
                "retrieval_notes": "Typhoon analysis requested refusal.",
            }
        )
        return base

    matched_columns = resolve_target_columns(analysis.get("target_fields", []))
    unknown_fields = find_unknown_target_fields(analysis.get("target_fields", []), df)
    if unknown_fields:
        base.update(
            {
                "retrieval_status": "refusal",
                "refusal_type": "field_not_in_directory",
                "confidence": 1.0,
                "retrieval_notes": f"Unknown target fields: {unknown_fields}",
            }
        )
        return base

    filtered, notes, confidence = filter_dataframe(df, analysis)
    matched_columns = [column for column in matched_columns if column in df.columns]

    if analysis.get("requires_count"):
        base.update(
            {
                "matched_rows": rows_to_records(filtered, limit=MAX_EVIDENCE_ROWS),
                "matched_columns": matched_columns,
                "computed_value": int(len(filtered)),
                "confidence": confidence if len(filtered) else 0.35,
                "retrieval_status": "found",
                "retrieval_notes": notes or "Computed count from deterministic filters.",
            }
        )
        return base

    if filtered.empty:
        base.update({"retrieval_notes": notes or "No deterministic match from analysis entities."})
        if analysis.get("entities", {}).get("person_name"):
            base["retrieval_status"] = "refusal"
            base["refusal_type"] = "person_not_found"
            base["confidence"] = 0.8
        return base

    if matched_columns and all(_all_blank(filtered, column) for column in matched_columns):
        if any(column in {"Nickname Thai", "Nickname English"} for column in matched_columns):
            refusal_type = "field_exists_but_blank"
        else:
            refusal_type = "field_not_in_directory"
        base.update(
            {
                "matched_rows": rows_to_records(filtered, limit=MAX_EVIDENCE_ROWS),
                "matched_columns": matched_columns,
                "retrieval_status": "refusal",
                "refusal_type": refusal_type,
                "confidence": 0.9,
                "retrieval_notes": "Matched entity but requested field is blank.",
            }
        )
        return base

    status = "found" if len(filtered) == 1 or analysis.get("requires_list") else "ambiguous"
    base.update(
        {
            "matched_rows": rows_to_records(filtered, limit=MAX_EVIDENCE_ROWS),
            "matched_columns": matched_columns,
            "computed_value": None,
            "confidence": confidence,
            "retrieval_status": status,
            "retrieval_notes": notes or f"Matched {len(filtered)} rows from deterministic filters.",
        }
    )
    if len(filtered) > MAX_EVIDENCE_ROWS:
        base["retrieval_notes"] += f" Evidence truncated to {MAX_EVIDENCE_ROWS} rows."
    return base


def resolve_target_columns(fields: list[Any]) -> list[str]:
    columns: list[str] = []
    for raw in fields:
        key = normalize_text(raw).replace(" ", "_")
        columns.extend(FIELD_TO_COLUMNS.get(key, []))
    return list(dict.fromkeys(columns))


def find_unknown_target_fields(fields: list[Any], df: pd.DataFrame) -> list[str]:
    unknown: list[str] = []
    for raw in fields:
        key = normalize_text(raw).replace(" ", "_")
        if key in FIELD_TO_COLUMNS:
            continue
        if any(normalize_text(raw) == normalize_text(column) for column in df.columns):
            continue
        if raw:
            unknown.append(str(raw))
    return unknown


def filter_dataframe(df: pd.DataFrame, analysis: dict[str, Any]) -> tuple[pd.DataFrame, str, float]:
    entities = analysis.get("entities") or {}
    candidates = df.copy()
    notes: list[str] = []
    confidence = 0.5

    person_name = entities.get("person_name")
    if person_name:
        person_matches = match_person_name(df, str(person_name))
        if not person_matches.empty:
            candidates = person_matches
            notes.append("Matched person_name.")
            confidence = 0.9 if len(candidates) == 1 else 0.75
        else:
            return df.iloc[0:0], "No person_name match.", 0.0

    nickname = entities.get("nickname")
    if nickname:
        candidates = match_columns(candidates, str(nickname), ["Nickname Thai", "Nickname English"])
        notes.append("Filtered by nickname.")
        confidence = max(confidence, 0.75 if len(candidates) else 0.0)

    for entity_key, column in ENTITY_FIELD_TO_COLUMN.items():
        value = entities.get(entity_key)
        if not value:
            continue
        columns = [column]
        if entity_key == "position":
            columns = ["Position in English", "Position in Thai"]
        candidates = match_columns(candidates, str(value), columns)
        notes.append(f"Filtered by {entity_key}.")
        confidence = max(confidence, 0.7 if len(candidates) else 0.0)

    if not any(entities.get(key) for key in ["person_name", "nickname", *ENTITY_FIELD_TO_COLUMN.keys()]):
        return df.iloc[0:0], "No usable entities in question analysis.", 0.0

    return candidates, " ".join(notes), confidence


def match_person_name(df: pd.DataFrame, query: str) -> pd.DataFrame:
    query_norm = normalize_text(query)
    if not query_norm:
        return df.iloc[0:0]

    full_english = (df["First Name English"].astype(str) + " " + df["Last Name English"].astype(str)).map(normalize_text)
    full_thai = (df["First Name Thai"].astype(str) + " " + df["Last Name Thai"].astype(str)).map(normalize_text)
    exact_mask = (full_english == query_norm) | (full_thai == query_norm)
    if exact_mask.any():
        return df[exact_mask]

    contains_mask = full_english.str.contains(re.escape(query_norm), regex=True, na=False) | full_thai.str.contains(
        re.escape(query_norm), regex=True, na=False
    )
    if contains_mask.any():
        return df[contains_mask]

    scores = full_english.map(lambda value: fuzz.token_set_ratio(query_norm, value)).combine(
        full_thai.map(lambda value: fuzz.token_set_ratio(query_norm, value)),
        max,
    )
    return df[scores >= 92]


def match_columns(df: pd.DataFrame, query: str, columns: list[str]) -> pd.DataFrame:
    query_norm = normalize_text(query)
    if not query_norm:
        return df.iloc[0:0]
    mask = pd.Series(False, index=df.index)
    for column in columns:
        if column not in df.columns:
            continue
        values = df[column].astype(str).map(normalize_text)
        exact = values == query_norm
        contains = values.str.contains(re.escape(query_norm), regex=True, na=False)
        fuzzy = values.map(lambda value: fuzz.token_set_ratio(query_norm, value) >= 94)
        mask = mask | exact | contains | fuzzy
    return df[mask]


def _all_blank(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].astype(str).str.strip().eq("").all()


def rows_to_records(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    return df.head(limit).fillna("").to_dict(orient="records")

