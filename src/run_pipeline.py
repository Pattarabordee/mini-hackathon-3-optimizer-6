from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .answer_generator import generate_answers
from .config import ConfigError, ensure_runtime_dirs, load_config
from .error_analysis import build_error_analysis
from .load_data import ensure_required_files
from .profile_data import build_data_profile
from .question_router import analyze_questions
from .retriever import retrieve_all
from .validator import validate_submission


def run_full_pipeline(force_analysis: bool = False, force_generation: bool = False) -> int:
    started = time.time()
    config = load_config()
    ensure_runtime_dirs(config)
    ensure_required_files(config.root)

    data_profile_path = build_data_profile(config.root)
    analyses, analysis_api_calls = analyze_questions(config, force=force_analysis)
    evidence = retrieve_all(config.root, analyses)
    answers, generation_api_calls = generate_answers(config, analyses, evidence, force=force_generation)
    validation_report = validate_submission(config.root)
    grade_stdout, grade_stderr, grade_returncode = run_local_grade(config.root)
    error_report_path = build_error_analysis(config.root, grade_stdout, grade_stderr)

    total_api_calls = analysis_api_calls + generation_api_calls
    run_report_path = write_run_report(
        root=config.root,
        data_profile_path=data_profile_path,
        validation_report=validation_report,
        grade_stdout=grade_stdout,
        grade_stderr=grade_stderr,
        grade_returncode=grade_returncode,
        total_api_calls=total_api_calls,
        mock_mode=config.use_mock_typhoon,
        elapsed_seconds=time.time() - started,
        answer_count=len(answers),
        error_report_path=error_report_path,
    )

    print_summary(validation_report, grade_stdout, total_api_calls, config.use_mock_typhoon, run_report_path)
    return 0 if validation_report["status"] == "pass" else 1


def run_validation_only() -> int:
    config = load_config()
    ensure_runtime_dirs(config)
    ensure_required_files(config.root)
    report = validate_submission(config.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


def run_grading_only() -> int:
    config = load_config()
    ensure_required_files(config.root)
    stdout, stderr, returncode = run_local_grade(config.root)
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    return returncode


def run_local_grade(root: Path) -> tuple[str, str, int]:
    submission = root / "outputs" / "submission.csv"
    if not submission.exists():
        return "", f"Missing submission file: {submission}", 1
    result = subprocess.run(
        [sys.executable, "grade.py", str(submission), "train_labels.json"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout, result.stderr, result.returncode


def write_run_report(
    root: Path,
    data_profile_path: Path,
    validation_report: dict,
    grade_stdout: str,
    grade_stderr: str,
    grade_returncode: int,
    total_api_calls: int,
    mock_mode: bool,
    elapsed_seconds: float,
    answer_count: int,
    error_report_path: Path,
) -> Path:
    path = root / "reports" / "run_report.md"
    lines = [
        "# Run Report",
        "",
        f"- Mock mode: {mock_mode}",
        f"- Typhoon real API calls this run: {total_api_calls}",
        f"- Answers generated/loaded: {answer_count}",
        f"- Elapsed seconds: {elapsed_seconds:.2f}",
        f"- Data profile: `{data_profile_path}`",
        f"- Error analysis: `{error_report_path}`",
        "",
        "## Validator",
        "",
        f"- Status: {validation_report.get('status')}",
        f"- Errors: {validation_report.get('summary', {}).get('error_count')}",
        f"- Warnings: {validation_report.get('summary', {}).get('warning_count')}",
        "",
        "## Local Grade",
        "",
        f"- Return code: {grade_returncode}",
        "",
        "```text",
        grade_stdout.strip() or "(no stdout captured)",
        "```",
        "",
    ]
    if grade_stderr.strip():
        lines.extend(["## Local Grade stderr", "", "```text", grade_stderr.strip(), "```", ""])
    if mock_mode:
        lines.extend(
            [
                "## Mock Mode Notice",
                "",
                "The generated submission is for pipeline testing only and is not a real Typhoon competition submission.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_summary(
    validation_report: dict,
    grade_stdout: str,
    total_api_calls: int,
    mock_mode: bool,
    run_report_path: Path,
) -> None:
    score_line = next((line for line in grade_stdout.splitlines() if line.startswith("Passed:")), "Passed: n/a")
    print("Pipeline complete")
    print(f"Validator status: {validation_report.get('status')}")
    print(score_line)
    print(f"Typhoon real API calls this run: {total_api_calls}")
    print(f"Mock mode: {mock_mode}")
    print(f"Run report: {run_report_path}")


def main() -> int:
    _configure_console_encoding()
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--grade-only", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument("--force-generation", action="store_true")
    args = parser.parse_args()

    try:
        if args.validate_only:
            return run_validation_only()
        if args.grade_only:
            return run_grading_only()
        return run_full_pipeline(force_analysis=args.force_analysis, force_generation=args.force_generation)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
