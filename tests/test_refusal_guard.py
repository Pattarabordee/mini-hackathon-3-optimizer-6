from src.refusal_guard import assert_no_refusal_leak, refusal_phrase


def test_canonical_refusal_phrase_exact_match():
    assert refusal_phrase("th", "field_not_in_directory") == "ไม่สามารถให้ข้อมูลนี้ได้"
    assert refusal_phrase("en", "person_not_found") == "no record found"


def test_refusal_does_not_leak_employee_id_or_extension():
    assert_no_refusal_leak(refusal_phrase("th", "prompt_injection"))
    assert_no_refusal_leak(refusal_phrase("en", "external_company"))

