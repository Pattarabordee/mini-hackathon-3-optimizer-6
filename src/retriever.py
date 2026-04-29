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

LOOKUP_COLUMNS = [
    "Employee ID",
    "Email Address",
    "Phone Extension",
    "Mobile No.",
    "First Name Thai",
    "Last Name Thai",
    "First Name English",
    "Last Name English",
    "Nickname Thai",
    "Nickname English",
    "Position in Thai",
    "Position in English",
    "Department",
    "Section",
    "Unit",
    "Office Location",
    "Branch",
    "Start Year",
    "Position Level",
]

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

    person_name = entities.get("person_name")
    if person_name:
        used_entity = True
        person_matches = match_person_name(df, str(person_name))
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
        if entity_key == "position":
            candidates = match_position_entity(candidates, str(value))
        else:
            candidates = match_columns(candidates, str(value), [column])
        notes.append(f"Filtered by {entity_key}.")
        confidence = max(confidence, 0.7 if len(candidates) else 0.0)

    other_values = normalize_other_entities(entities.get("other"))
    if other_values:
        other_matches, other_note, other_confidence = match_lookup_values(df, other_values)
        if not other_matches.empty:
            used_entity = True
            if candidates.empty or not notes:
                candidates = other_matches
            elif analysis.get("requires_list") or len(other_values) > 1:
                candidates = merge_rows(candidates, other_matches)
            else:
                narrowed = candidates[candidates.index.isin(other_matches.index)]
                candidates = narrowed if not narrowed.empty else other_matches
            notes.append(other_note)
            confidence = max(confidence, other_confidence)

    if not used_entity:
        fallback, fallback_note, fallback_confidence = fallback_match_from_question(df, analysis)
        if not fallback.empty:
            return fallback, fallback_note, fallback_confidence
        return df.iloc[0:0], "No usable entities in question analysis.", 0.0

    if candidates.empty:
        fallback, fallback_note, fallback_confidence = fallback_match_from_question(df, analysis)
        if not fallback.empty:
            return fallback, f"Entity filters were empty. {fallback_note}", fallback_confidence

    if analysis.get("requires_count") or analysis.get("requires_list"):
        refined, refined_note, refined_confidence = fallback_group_match_from_question(df, analysis)
        if not refined.empty and (candidates.empty or len(refined) < len(candidates)):
            return refined, f"Refined broad group filter. {refined_note}", max(confidence, refined_confidence)

    return candidates, " ".join(notes), confidence


def fallback_match_from_question(df: pd.DataFrame, analysis: dict[str, Any]) -> tuple[pd.DataFrame, str, float]:
    """Deterministic rescue pass using exact directory values found in the question text."""
    raw_question = str(analysis.get("question", ""))
    question = normalize_text(raw_question)
    if not question:
        return df.iloc[0:0], "No question text for fallback retrieval.", 0.0

    requires_group_result = bool(analysis.get("requires_count") or analysis.get("requires_list"))
    lookup_values = extract_lookup_values(raw_question)
    if lookup_values:
        matched, note, confidence = match_lookup_values(df, lookup_values)
        if not matched.empty:
            return matched, f"Fallback matched extracted lookup token. {note}", confidence

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


def fallback_group_match_from_question(df: pd.DataFrame, analysis: dict[str, Any]) -> tuple[pd.DataFrame, str, float]:
    question = normalize_text(analysis.get("question", ""))
    if not question:
        return df.iloc[0:0], "No question text for group refinement.", 0.0
    for column in [
        "Unit",
        "Section",
        "Department",
        "Branch",
        "Office Location",
        "Position Level",
        "Start Year",
        "Position in English",
        "Position in Thai",
    ]:
        matched = _match_question_against_column(df, question, column, min_len=2)
        if not matched.empty:
            return matched, f"Group refinement matched `{column}` from question text.", 0.66
    return df.iloc[0:0], "Group refinement found no exact directory value in question text.", 0.0


def normalize_other_entities(raw: Any) -> list[str]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    normalized: list[str] = []
    for value in values:
        if isinstance(value, dict):
            value = " ".join(str(item) for item in value.values() if item)
        text = str(value or "").strip()
        if text:
            normalized.append(text)
    return list(dict.fromkeys(normalized))


def extract_lookup_values(raw_question: str) -> list[str]:
    values: list[str] = []
    values.extend(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", raw_question, flags=re.IGNORECASE))

    for candidate in re.findall(r"(?:\d[\s\-().]*){5,12}", raw_question):
        digits = re.sub(r"\D", "", candidate)
        if len(digits) in {5, 8, 9, 10}:
            values.append(digits)

    code_pattern = r"\b[A-Z]{2,}[A-Z0-9]{1,}(?:[-_][A-Z0-9]+)*\b|\b[A-Za-z]{2,}[A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b"
    stopwords = {
        "AND",
        "ARE",
        "FOR",
        "FROM",
        "HOW",
        "THE",
        "WHAT",
        "WHEN",
        "WHERE",
        "WHICH",
        "WHO",
        "WITH",
    }
    for token in re.findall(code_pattern, raw_question):
        if token.upper() not in stopwords:
            values.append(token)

    values.extend(re.findall(r"\b[A-Z]?[a-z]+(?:[A-Z][a-z]+)+\b", raw_question))

    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def match_lookup_values(df: pd.DataFrame, values: list[str]) -> tuple[pd.DataFrame, str, float]:
    frames: list[pd.DataFrame] = []
    matched_columns: list[str] = []

    for value in values:
        person_matches = match_person_name(df, value)
        if not person_matches.empty:
            frames.append(person_matches)
            matched_columns.extend(["First Name Thai", "Last Name Thai", "First Name English", "Last Name English"])

        column_matches, columns = match_value_across_columns(df, value, LOOKUP_COLUMNS)
        if not column_matches.empty:
            frames.append(column_matches)
            matched_columns.extend(columns)

    if not frames:
        return df.iloc[0:0], "Lookup values did not match directory columns.", 0.0

    merged = frames[0]
    for frame in frames[1:]:
        merged = merge_rows(merged, frame)
    columns = list(dict.fromkeys(matched_columns))
    confidence = 0.86 if len(merged) <= max(len(values), 1) else 0.68
    return merged, f"Matched lookup values against columns: {columns}.", confidence


def match_value_across_columns(df: pd.DataFrame, value: str, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    variants = lookup_value_variants(value)
    if not variants:
        return df.iloc[0:0], []

    query_digits = re.sub(r"\D", "", value)
    exact_mask = pd.Series(False, index=df.index)
    contains_mask = pd.Series(False, index=df.index)
    matched_columns: list[str] = []

    for column in columns:
        if column not in df.columns:
            continue
        raw_values = df[column].astype(str)
        normalized_values = raw_values.map(normalize_text)
        column_exact = pd.Series(False, index=df.index)
        for variant in variants:
            column_exact = column_exact | (normalized_values == variant)

        if query_digits and column in {"Employee ID", "Phone Extension", "Mobile No."}:
            digit_values = raw_values.map(lambda item: re.sub(r"\D", "", item))
            column_exact = column_exact | (digit_values == query_digits)

        if column_exact.any():
            exact_mask = exact_mask | column_exact
            matched_columns.append(column)
            continue

        contain_variants = [variant for variant in variants if len(variant) >= 4]
        if contain_variants:
            column_contains = pd.Series(False, index=df.index)
            for variant in contain_variants:
                column_contains = column_contains | normalized_values.str.contains(re.escape(variant), regex=True, na=False)
            if query_digits and column in {"Employee ID", "Phone Extension", "Mobile No."}:
                digit_values = raw_values.map(lambda item: re.sub(r"\D", "", item))
                column_contains = column_contains | digit_values.str.contains(re.escape(query_digits), regex=True, na=False)
            if column_contains.any():
                contains_mask = contains_mask | column_contains
                matched_columns.append(column)

    mask = exact_mask if exact_mask.any() else contains_mask
    return df[mask], list(dict.fromkeys(matched_columns))


def lookup_value_variants(value: str) -> list[str]:
    text = str(value or "").strip()
    normalized = normalize_text(text)
    variants = [normalized]

    compact = re.sub(r"[^A-Za-z0-9]+", "", text)
    if compact:
        variants.append(normalize_text(compact))

    camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", text)
    if len(camel_parts) > 1:
        variants.append("".join(part[0] for part in camel_parts if part).casefold())

    words = normalized.split()
    if len(words) > 1:
        variants.append("".join(word[0] for word in words if word))

    return list(dict.fromkeys(variant for variant in variants if variant))


def merge_rows(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([left, right]).loc[lambda frame: ~frame.index.duplicated(keep="first")]


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


def match_position_entity(df: pd.DataFrame, query: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant in position_query_variants(query):
        title_matches = match_columns(df, variant, ["Position in English", "Position in Thai"])
        if not title_matches.empty:
            frames.append(title_matches)
        code_matches, _ = match_value_across_columns(df, variant, ["Unit", "Section", "Department", "Position Level"])
        if not code_matches.empty:
            frames.append(code_matches)

    if not frames:
        return df.iloc[0:0]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merge_rows(merged, frame)
    return merged


def position_query_variants(query: str) -> list[str]:
    expansions = {
        "ceo": "chief executive officer",
        "cfo": "chief financial officer",
        "cto": "chief technology officer",
        "coo": "chief operating officer",
        "cmo": "chief marketing officer",
        "cpo": "chief product officer",
        "chro": "chief human resources officer",
        "evp": "executive vice president",
        "svp": "senior vice president",
        "avp": "assistant vice president",
        "vp": "vice president",
        "ea": "executive assistant",
        "secretary": "executive assistant",
    }
    normalized = normalize_text(query)
    variants = [query, normalized]
    tokens = normalized.split()
    expanded_tokens = [expansions.get(token, token) for token in tokens]
    expanded = " ".join(expanded_tokens).strip()
    if expanded and expanded != normalized:
        variants.append(expanded)

    compact = re.sub(r"[^A-Za-z0-9]+", "", query)
    if compact and compact.casefold() in expansions:
        variants.append(expansions[compact.casefold()])

    return list(dict.fromkeys(value for value in variants if value))


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
