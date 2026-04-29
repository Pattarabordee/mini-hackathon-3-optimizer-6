from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import AppConfig
from .jsonl_utils import append_jsonl


@dataclass
class TyphoonResult:
    ok: bool
    content: str
    error: str | None = None
    mock: bool = False
    api_call_made: bool = False


class TyphoonClient:
    """OpenAI-compatible Chat Completions client for the required Typhoon model."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.real_api_calls = 0

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> TyphoonResult:
        if self.config.use_mock_typhoon:
            return TyphoonResult(
                ok=True,
                content=self._mock_response(messages),
                mock=True,
                api_call_made=False,
            )

        payload = {
            "model": self.config.typhoon_model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.config.typhoon_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.config.typhoon_base_url}/chat/completions"

        last_error: str | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                self.real_api_calls += 1
                append_jsonl(
                    self.config.cache_dir / "typhoon_api_calls.jsonl",
                    {
                        "ts": time.time(),
                        "model": self.config.typhoon_model,
                        "base_url": self.config.typhoon_base_url,
                        "status_code": response.status_code,
                        "attempt": attempt + 1,
                    },
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return TyphoonResult(ok=True, content=content, api_call_made=True)
            except Exception as exc:  # noqa: BLE001 - converted to structured error for caller
                last_error = str(exc)
                if attempt < self.config.max_retries:
                    time.sleep(2**attempt)

        return TyphoonResult(ok=False, content="", error=last_error, api_call_made=True)

    def _mock_response(self, messages: list[dict[str, str]]) -> str:
        joined = "\n".join(message.get("content", "") for message in messages)
        if "QUESTION_ANALYSIS_SCHEMA" in joined:
            payload = _extract_last_json_object(joined)
            return json.dumps(
                {
                    "id": payload.get("id", ""),
                    "language": payload.get("language", "en"),
                    "question": payload.get("question", ""),
                    "intent": "unknown",
                    "entities": {
                        "person_name": None,
                        "nickname": None,
                        "position": None,
                        "department": None,
                        "section": None,
                        "unit": None,
                        "office": None,
                        "branch": None,
                        "field": None,
                        "other": [],
                    },
                    "target_fields": [],
                    "needs_refusal": False,
                    "refusal_type": None,
                    "requires_count": False,
                    "requires_list": False,
                    "requires_exact_count": False,
                    "notes": "MOCK_TYPHOON: not real question analysis.",
                },
                ensure_ascii=False,
            )

        if "ANSWER_GENERATION_SCHEMA" in joined:
            payload = _extract_last_json_object(joined)
            language = payload.get("language", "en")
            response = (
                "MOCK_TYPHOON: ไม่ใช่คำตอบจริงจาก Typhoon"
                if language == "th"
                else "MOCK_TYPHOON: not a real Typhoon answer"
            )
            return json.dumps({"response": response}, ensure_ascii=False)

        return json.dumps({"response": "MOCK_TYPHOON: unsupported mock prompt"}, ensure_ascii=False)


def _extract_last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    best: dict[str, Any] = {}
    language_payload: dict[str, Any] = {}
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            best = value
            if "language" in value:
                language_payload = value
    return language_payload or best
