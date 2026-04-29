from pathlib import Path

from src.answer_generator import deterministic_answer, generate_one
from src.config import AppConfig


class RaisingClient:
    real_api_calls = 0

    def chat(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Typhoon should not be called when retrieval has no evidence")


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        root=tmp_path,
        typhoon_api_key="test-key",
        typhoon_base_url="https://api.opentyphoon.ai/v1",
        typhoon_model="typhoon-v2.5-30b-a3b-instruct",
        mock_typhoon=False,
        timeout_seconds=1,
        max_retries=0,
        cache_dir=tmp_path / "cache",
        outputs_dir=tmp_path / "outputs",
        reports_dir=tmp_path / "reports",
    )


def test_no_evidence_skips_typhoon_and_uses_language_fallback(tmp_path):
    record = generate_one(
        _config(tmp_path),
        RaisingClient(),
        {"id": "q1", "language": "th"},
        {"retrieval_status": "not_found"},
    )

    assert record["source"] == "no_evidence_fallback"
    assert record["response"] == "ไม่พบข้อมูล"


def test_ambiguous_field_answers_include_requested_values():
    response = deterministic_answer(
        {"id": "q2", "language": "en", "requires_list": False},
        {
            "retrieval_status": "ambiguous",
            "matched_columns": ["Phone Extension"],
            "matched_rows": [
                {
                    "First Name English": "Somchai",
                    "Last Name English": "Jaidee",
                    "First Name Thai": "",
                    "Last Name Thai": "",
                    "Phone Extension": "12345",
                },
                {
                    "First Name English": "Jane",
                    "Last Name English": "Finance",
                    "First Name Thai": "",
                    "Last Name Thai": "",
                    "Phone Extension": "54321",
                },
            ],
        },
    )

    assert "Somchai Jaidee: 12345" in response
    assert "Jane Finance: 54321" in response
