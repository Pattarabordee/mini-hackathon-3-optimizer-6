from __future__ import annotations

from pathlib import Path

import pandas as pd

from .load_data import read_employees


SENSITIVE_COLUMNS = {
    "Employee ID",
    "First Name Thai",
    "Last Name Thai",
    "First Name English",
    "Last Name English",
    "Nickname Thai",
    "Nickname English",
    "Email Address",
    "Phone Extension",
    "Mobile No.",
    "Office Location",
}


COLUMN_HINTS = {
    "Employee ID": ["Employee ID"],
    "Thai name": ["First Name Thai", "Last Name Thai"],
    "English name": ["First Name English", "Last Name English"],
    "Thai nickname": ["Nickname Thai"],
    "English nickname": ["Nickname English"],
    "Thai position": ["Position in Thai"],
    "English position": ["Position in English"],
    "department": ["Department"],
    "section": ["Section"],
    "unit": ["Unit"],
    "email": ["Email Address"],
    "phone extension": ["Phone Extension"],
    "mobile": ["Mobile No."],
    "office": ["Office Location"],
    "branch": ["Branch"],
    "start year": ["Start Year"],
}


IMPORTANT_CATEGORICAL_COLUMNS = [
    "Department",
    "Section",
    "Unit",
    "Branch",
    "Position Level",
    "Start Year",
    "Position in English",
    "Position in Thai",
]


def _redacted_sample(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    sample = df.head(n).copy()
    for column in sample.columns:
        if column in SENSITIVE_COLUMNS:
            sample[column] = sample[column].map(lambda value: "<redacted>" if str(value).strip() else "")
    return sample


def build_data_profile(root: Path) -> Path:
    df = read_employees(root)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "data_profile.md"

    missing = df.replace("", pd.NA).isna().sum()
    unique = df.nunique(dropna=True)

    lines: list[str] = [
        "# Data Profile",
        "",
        "This is a technical profile for retrieval engineering. Sensitive sample values are redacted.",
        "",
        "## Shape",
        "",
        f"- Rows: {len(df)}",
        f"- Columns: {len(df.columns)}",
        "",
        "## Columns",
        "",
        *[f"- `{column}`" for column in df.columns],
        "",
        "## Redacted Sample Rows",
        "",
        _redacted_sample(df).to_markdown(index=False),
        "",
        "## Missing Values",
        "",
        "| Column | Missing | Non-empty | Unique |",
        "|---|---:|---:|---:|",
    ]

    for column in df.columns:
        lines.append(
            f"| `{column}` | {int(missing[column])} | {len(df) - int(missing[column])} | {int(unique[column])} |"
        )

    lines.extend(["", "## Important Categorical Columns", ""])
    for column in IMPORTANT_CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue
        counts = df[column].replace("", pd.NA).dropna().value_counts()
        lines.append(f"### `{column}`")
        lines.append("")
        lines.append(f"- Unique values: {int(counts.shape[0])}")
        lines.append(f"- Top group sizes: {', '.join(str(int(value)) for value in counts.head(10).tolist())}")
        lines.append("")

    lines.extend(["## Column Role Detection", ""])
    for role, columns in COLUMN_HINTS.items():
        present = [column for column in columns if column in df.columns]
        lines.append(f"- {role}: {', '.join(f'`{column}`' for column in present) if present else 'not found'}")

    nickname_missing = {}
    for column in ["Nickname Thai", "Nickname English"]:
        if column in df.columns:
            nickname_missing[column] = int(missing[column])

    lines.extend(
        [
            "",
            "## Retrieval Observations",
            "",
            "- `Employee ID` and `Email Address` are unique when present.",
            "- Full Thai and English names should be preferred over first-name-only matching.",
            "- Nickname lookup should support ambiguity because nicknames are not complete and may repeat.",
            "- Department and section can be blank for some employees; retrieval should signal blank fields clearly.",
            "- Directory fields contain sensitive contact data; reports and Git history must avoid raw values.",
            "",
        ]
    )

    if nickname_missing:
        lines.append("## Nickname Missing Values")
        lines.append("")
        for column, count in nickname_missing.items():
            lines.append(f"- `{column}` missing rows: {count}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(build_data_profile(root))


if __name__ == "__main__":
    main()

