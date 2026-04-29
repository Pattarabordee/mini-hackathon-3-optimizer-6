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
    "person_name": ["First Name Thai", "Last Name Thai", "First Name English", "Last Name English"],
    "thai_name": ["First Name Thai", "Last Name Thai"],
    "english_name": ["First Name English", "Last Name English"],
    "nickname": ["Nickname Thai", "Nickname English"],
    "thai_nickname": ["Nickname Thai"],
    "english_nickname": ["Nickname English"],
    "position": ["Position in Thai", "Position in English"],
    "thai_position": ["Position in Thai"],
    "english_position": ["Position in English"],
    "department": ["Department"],
    "section": ["Section"],
    "unit": ["Unit"],
    "office": ["Office Location"],
    "location": ["Office Location"],
    "branch": ["Branch"],
    "email": ["Email Address"],
    "contact": ["Email Address", "Phone Extension", "Mobile No."],
    "contact_info": ["Email Address", "Phone Extension", "Mobile No."],
    "contact_number": ["Phone Extension", "Mobile No."],
    "phone_number": ["Phone Extension", "Mobile No."],
    "phone": ["Phone Extension"],
    "extension": ["Phone Extension"],
    "phone_extension": ["Phone Extension"],
    "contact_id": ["Phone Extension"],
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
    used_entity = False

    filtered_by_schema, filter_notes, filter_confidence = apply_schema_filters(df, analysis.get("filters") or [])
    if filter_notes:
        candidates = filtered_by_schema
        notes.append(filter_notes)
        confidence = filter_confidence
        used_entity = True

    person_name = entities.get("person_name")
    if person_name:
        used_entity = True
        person_matches = match_person_name(candidates if not candidates.empty else df, str(person_name))
        if not person_matches.empty:
            candidates = person_matches
            notes.append("Matched person_name.")
            confidence = 0.9 if len(candidates) == 1 else 0.75
        else:
            fallback, fallback_note, fallback_confidence = fallback_match_from_question(df, analysis)
            if not fallback.empty:
                return fallback, f"No person_name match. {fallback_note}", fallback_confidence
            return df.iloc[0:0], "No person_name match.", 0.0

    nickname = entities.get("nickname")
    if nickname:
        used_entity = True
        candidates = match_columns(candidates, str(nickname), ["Nickname Thai", "Nickname English"])
        notes.append("Filtered by nickname.")
        confidence = max(confidence, 0.75 if len(candidates) else 0.0)

    for entity_key, column in ENTITY_FIELD_TO_COLUMN.items():
        value = entities.get(entity_key)
        if not value:
            continue
        used_entity = True
        columns = [column]
        if entity_key == "position":
            columns = ["Position in English", "Position in Thai"]
        candidates = match_columns(candidates, str(value), columns)
        notes.append(f"Filtered by {entity_key}.")
        confidence = max(confidence, 0.7 if len(candidates) else 0.0)

    if not used_entity:
        fallback, fallback_note, fallback_confidence = fallback_match_from_question(df, analysis)
        if not fallback.empty:
            return fallback, fallback_note, fallback_confidence
        return df.iloc[0:0], "No usable entities in question analysis.", 0.0

    if candidates.empty:
        fallback, fallback_note, fallback_confidence = fallback_match_from_question(df, analysis)
        if not fallback.empty:
            return fallback, f"Entity filters were empty. {fallback_note}", fallback_confidence

    return candidates, " ".join(notes), confidence


def apply_schema_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> tuple[pd.DataFrame, str, float]:
    candidates = df.copy()
    notes: list[str] = []
    confidence = 0.0
    for item in filters:
        field = str(item.get("field") or "").strip()
        value = str(item.get("value") or "").strip()
        if not field or not value:
            continue
        if field in {"name", "thai_name", "english_name"}:
            matched = match_person_name(candidates, value)
        else:
            columns = FIELD_TO_COLUMNS.get(field, [])
            matched = match_columns(candidates, value, columns)
        if matched.empty:
            continue
        candidates = matched
        notes.append(f"Applied filter {field}.")
        confidence = max(confidence, 0.78 if len(candidates) == 1 else 0.68)
    if not notes:
        return df.iloc[0:0], "", 0.0
    return candidates, " ".join(notes), confidence


def fallback_match_from_question(df: pd.DataFrame, analysis: dict[str, Any]) -> tuple[pd.DataFrame, str, float]:
    """Deterministic rescue pass using exact directory values found in the question text."""
    question = normalize_text(analysis.get("question", ""))
    if not question:
        return df.iloc[0:0], "No question text for fallback retrieval.", 0.0

    requires_group_result = bool(analysis.get("requires_count") or analysis.get("requires_list"))

    full_english = (df["First Name English"].astype(str) + " " + df["Last Name English"].astype(str)).map(normalize_text)
    full_thai = (df["First Name Thai"].astype(str) + " " + df["Last Name Thai"].astype(str)).map(normalize_text)
    for label, values in [("full English name", full_english), ("full Thai name", full_thai)]:
        mask = values.map(lambda value: bool(value) and len(value) >= 5 and value in question)
        if mask.any():
            return df[mask], f"Fallback matched {label} from question text.", 0.82

    strong_columns = [
        "Email Address",
        "Phone Extension",
        "Mobile No.",
        "Unit",
        "Section",
        "Position in English",
        "Position in Thai",
        "Nickname English",
        "Nickname Thai",
    ]
    for column in strong_columns:
        matched = _match_question_against_column(df, question, column, min_len=3)
        if not matched.empty:
            return matched, f"Fallback matched `{column}` from question text.", 0.72

    if requires_group_result:
        for column in ["Department", "Branch", "Office Location", "Position Level", "Start Year"]:
            matched = _match_question_against_column(df, question, column, min_len=2)
            if not matched.empty:
                return matched, f"Fallback matched group column `{column}` from question text.", 0.62

    return df.iloc[0:0], "Fallback retrieval found no exact directory value in question text.", 0.0


def _match_question_against_column(df: pd.DataFrame, question: str, column: str, min_len: int) -> pd.DataFrame:
    if column not in df.columns:
        return df.iloc[0:0]
    values = df[column].astype(str).map(normalize_text)
    unique_values = sorted({value for value in values if len(value) >= min_len}, key=len, reverse=True)
    padded_question = f" {question} "
    for value in unique_values:
        padded_value = f" {value} "
        if padded_value in padded_question or (len(value) >= 5 and value in question):
            return df[values == value]
    return df.iloc[0:0]


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
