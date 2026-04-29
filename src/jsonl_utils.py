from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path, key: str | None = None) -> list[dict[str, Any]] | dict[str, dict[str, Any]]:
    if not path.exists():
        return {} if key else []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if key is None:
        return records
    return {str(record.get(key)): record for record in records if record.get(key) is not None}


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

