from __future__ import annotations

import re


CANONICAL_REFUSALS = {
    "th": {
        "field_not_in_directory": "ไม่สามารถให้ข้อมูลนี้ได้",
        "person_not_found": "ไม่พบข้อมูล",
        "speculation_or_opinion": "ไม่สามารถให้ความเห็นได้",
        "external_company": "ไม่ใช่ข้อมูลของฟ้าใหม่",
        "prompt_injection": "ขอปฏิเสธคำขอ",
        "field_exists_but_blank": "ไม่มีชื่อเล่นในระบบ",
    },
    "en": {
        "field_not_in_directory": "cannot provide this information",
        "person_not_found": "no record found",
        "speculation_or_opinion": "cannot offer an opinion",
        "external_company": "not a FahMai record",
        "prompt_injection": "request declined",
        "field_exists_but_blank": "nickname not listed",
    },
}

REFUSAL_TYPES = {
    None,
    "field_not_in_directory",
    "person_not_found",
    "speculation_or_opinion",
    "external_company",
    "prompt_injection",
    "field_exists_but_blank",
}


EMPLOYEE_ID_PATTERN = re.compile(r"\b0[08]\d{6}\b")
PHONE_EXTENSION_PATTERN = re.compile(r"\b\d{5}\b")
MOBILE_PATTERN = re.compile(r"(?<!\d)(?:0\d{2}[-\s]?\d{3}[-\s]?\d{4}|\+66[-\s]?\d{8,9})(?!\d)")


def refusal_phrase(language: str, refusal_type: str) -> str:
    lang = "th" if language == "th" else "en"
    if refusal_type not in CANONICAL_REFUSALS[lang]:
        raise ValueError(f"Unknown refusal_type: {refusal_type}")
    return CANONICAL_REFUSALS[lang][refusal_type]


def should_refuse(analysis: dict, evidence: dict | None = None) -> tuple[bool, str | None]:
    refusal_type = analysis.get("refusal_type")
    if analysis.get("needs_refusal") and refusal_type:
        return True, refusal_type
    if evidence:
        status = evidence.get("retrieval_status")
        ev_type = evidence.get("refusal_type") or evidence.get("retrieval_status_reason")
        if status == "refusal" and ev_type:
            return True, ev_type
    return False, None


def apply_refusal(analysis: dict, evidence: dict | None = None) -> str | None:
    do_refuse, refusal_type = should_refuse(analysis, evidence)
    if not do_refuse or not refusal_type:
        return None
    answer = refusal_phrase(analysis.get("language", "en"), refusal_type)
    assert_no_refusal_leak(answer)
    return answer


def assert_no_refusal_leak(answer: str) -> None:
    if EMPLOYEE_ID_PATTERN.search(answer):
        raise ValueError("Refusal response leaks an Employee ID pattern")
    if PHONE_EXTENSION_PATTERN.search(answer):
        raise ValueError("Refusal response leaks a phone extension pattern")
    if MOBILE_PATTERN.search(answer):
        raise ValueError("Refusal response leaks a mobile number pattern")


def is_canonical_refusal(answer: str, language: str) -> bool:
    lang = "th" if language == "th" else "en"
    return answer in set(CANONICAL_REFUSALS[lang].values())

