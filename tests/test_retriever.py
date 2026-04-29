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

