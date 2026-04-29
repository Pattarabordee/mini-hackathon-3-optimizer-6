#!/usr/bin/env python3
"""Fail if protected data files or likely secrets are about to enter Git."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PROTECTED_PATHS = {
    "employees.csv",
    "questions.csv",
    "train_labels.json",
    "FahMai Directory Q&A.pdf",
}

PROTECTED_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".parquet",
    ".xlsx",
    ".xls",
}

SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    "aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "openai-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "google api key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "assigned secret": re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd)\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
    ),
}

PII_PATTERNS = {
    "email addresses": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone-like numbers": re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"),
}

IGNORED_DIRS = {
    ".git",
    ".git-store",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
}


def run_git(args: list[str]) -> list[str]:
    env = os.environ.copy()
    env.setdefault("GIT_OPTIONAL_LOCKS", "0")
    git_args = ["git"]
    if (ROOT / ".git-store" / "config").exists():
        git_args.extend(["--git-dir=.git-store", "--work-tree=."])
    result = subprocess.run(
        [*git_args, *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def tracked_files() -> list[Path]:
    files = run_git(["ls-files"])
    return [ROOT / name for name in files]


def working_tree_files() -> list[Path]:
    found: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        rel_parts = path.relative_to(ROOT).parts
        if any(part in IGNORED_DIRS for part in rel_parts):
            continue
        found.append(path)
    return found


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_protected_path(path: Path) -> bool:
    relative = rel(path)
    if relative in PROTECTED_PATHS or relative.startswith("data/"):
        return True
    return path.suffix.lower() in PROTECTED_SUFFIXES


def read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > 2_000_000:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def scan(paths: list[Path]) -> int:
    failures: list[str] = []
    warnings: list[str] = []

    for path in sorted({p.resolve() for p in paths if p.exists() and p.is_file()}):
        relative = rel(path)

        if is_protected_path(path):
            failures.append(f"{relative}: protected local/private data path")
            continue

        text = read_text(path)
        if text is None:
            continue

        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{relative}: likely {label}")

        for label, pattern in PII_PATTERNS.items():
            count = len(pattern.findall(text))
            if count >= 5:
                warnings.append(f"{relative}: contains {count} {label}")

    if failures:
        print("Sensitive-data scan failed:")
        for item in failures:
            print(f"  - {item}")

    if warnings:
        print("Sensitive-data scan warnings:")
        for item in warnings:
            print(f"  - {item}")

    if failures:
        print("\nMove these files outside Git or add a sanitized sample instead.")
        return 1

    print("Sensitive-data scan passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="scan the full working tree")
    args = parser.parse_args()

    paths = working_tree_files() if args.all else tracked_files()
    return scan(paths)


if __name__ == "__main__":
    sys.exit(main())
