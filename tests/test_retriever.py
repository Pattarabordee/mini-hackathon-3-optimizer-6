import pandas as pd

from src.retriever import retrieve_one


def _df():
    return pd.DataFrame(
        [
            {
                "Employee ID": "0800001",
                "Department": "TEC",
                "Section": "TEC-INF",
                "Unit": "TEC-INF-1",
                "Position in Thai": "วิศวกร",
                "Position in English": "Engineer",
                "First Name Thai": "สมชาย",
                "Last Name Thai": "ใจดี",
                "First Name English": "Somchai",
                "Last Name English": "Jaidee",
                "Nickname Thai": "ชัย",
                "Nickname English": "Chai",
                "Email Address": "somchai@example.com",
                "Phone Extension": "12345",
                "Mobile No.": "0812345678",
                "Office Location": "HQ",
                "Branch": "BKK",
                "Start Year": "2024",
                "Position Level": "IC",
            }
        ]
    )


def _exec_df():
    return pd.DataFrame(
        [
            {
                "Employee ID": "0800002",
                "Department": "FIN",
                "Section": "FIN-EXEC",
                "Unit": "CFO",
                "Position in Thai": "",
                "Position in English": "CHIEF FINANCIAL OFFICER",
                "First Name Thai": "",
                "Last Name Thai": "",
                "First Name English": "Jane",
                "Last Name English": "Finance",
                "Nickname Thai": "",
                "Nickname English": "",
                "Email Address": "jane@example.com",
                "Phone Extension": "54321",
                "Mobile No.": "0899999999",
                "Office Location": "HQ",
                "Branch": "BKK",
                "Start Year": "2024",
                "Position Level": "C-level",
            }
        ]
    )


def _group_df():
    base = _df().iloc[0].to_dict()
    second = {**base, "Employee ID": "0800003", "Section": "TEC-OPS", "Unit": "TEC-OPS-1", "First Name English": "Alice"}
    return pd.DataFrame([base, second])


def test_retriever_basic_exact_person_match():
    analysis = {
        "id": "t001",
        "intent": "lookup_contact",
        "entities": {"person_name": "Somchai Jaidee"},
        "target_fields": ["email"],
        "requires_count": False,
        "requires_list": False,
    }
    evidence = retrieve_one(_df(), analysis)
    assert evidence["retrieval_status"] == "found"
    assert evidence["matched_rows"][0]["Email Address"] == "somchai@example.com"


def test_retriever_not_found_when_person_missing():
    analysis = {
        "id": "t002",
        "intent": "lookup_person",
        "entities": {"person_name": "Missing Person"},
        "target_fields": [],
        "requires_count": False,
        "requires_list": False,
    }
    evidence = retrieve_one(_df(), analysis)
    assert evidence["retrieval_status"] == "refusal"
    assert evidence["refusal_type"] == "person_not_found"


def test_retriever_uses_other_entities_for_reverse_lookup():
    analysis = {
        "id": "t003",
        "intent": "lookup_person",
        "entities": {"other": ["12345"]},
        "target_fields": ["name"],
        "requires_count": False,
        "requires_list": False,
    }
    evidence = retrieve_one(_df(), analysis)
    assert evidence["retrieval_status"] == "found"
    assert evidence["matched_rows"][0]["First Name English"] == "Somchai"


def test_retriever_expands_position_acronyms():
    analysis = {
        "id": "t005",
        "intent": "lookup_person",
        "entities": {"position": "CFO"},
        "target_fields": ["name"],
        "requires_count": False,
        "requires_list": False,
    }
    evidence = retrieve_one(_exec_df(), analysis)
    assert evidence["retrieval_status"] == "found"
    assert evidence["matched_rows"][0]["First Name English"] == "Jane"


def test_retriever_refines_group_counts_from_question_text():
    analysis = {
        "id": "t006",
        "intent": "count_employees",
        "entities": {"department": "TEC"},
        "target_fields": [],
        "question": "How many people are in TEC-INF?",
        "requires_count": True,
        "requires_list": False,
    }
    evidence = retrieve_one(_group_df(), analysis)
    assert evidence["retrieval_status"] == "found"
    assert evidence["computed_value"] == 1


def test_retriever_matches_camel_case_org_name_to_code():
    analysis = {
        "id": "t007",
        "intent": "list_employees",
        "entities": {"other": ["TechCrew"]},
        "target_fields": ["name"],
        "requires_count": False,
        "requires_list": True,
    }
    df = _df().copy()
    df.loc[0, "Department"] = "TC"
    evidence = retrieve_one(df, analysis)
    assert evidence["retrieval_status"] == "found"
    assert evidence["matched_rows"][0]["Department"] == "TC"


def test_retriever_extracts_lookup_tokens_from_question_text():
    analysis = {
        "id": "t004",
        "intent": "lookup_person",
        "entities": {},
        "target_fields": ["name"],
        "question": "Who uses mobile 081-234-5678?",
        "requires_count": False,
        "requires_list": False,
    }
    evidence = retrieve_one(_df(), analysis)
    assert evidence["retrieval_status"] == "found"
    assert evidence["matched_rows"][0]["First Name English"] == "Somchai"
