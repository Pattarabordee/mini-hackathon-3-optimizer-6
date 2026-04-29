"""Kaggle grader for FahMai Directory Q&A.

Token-based pass/fail per item. Score = passes / total.

Usage:
  python grade.py submission.csv train_labels.json          # score against 158 public items
  python grade.py submission.csv ground_truth_full.json     # full 300-item score (server-side)

Inputs:
  - submission.csv:  columns `id,response` (UTF-8)
  - ground_truth: JSON with {"items": [{id, expected_answer, ...}, ...]}
"""
from __future__ import annotations
import csv, io, json, re, sys
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if len(sys.argv) < 3:
    print("Usage: python grade.py <submission.csv> <ground_truth.json>")
    sys.exit(2)

SUB_PATH, GT_PATH = sys.argv[1], sys.argv[2]


def load_submission(path: str) -> dict[str, str]:
    subs: dict[str, str] = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            subs[row["id"].strip()] = (row.get("response") or "").strip()
    return subs


def load_ground_truth(path: str) -> list[dict]:
    return json.loads(open(path, encoding="utf-8").read())["items"]


def grade_item(gt: dict, resp: str) -> tuple[bool, list[str]]:
    """Return (passed, reasons). Port of the in-repo grader contract."""
    ea = gt.get("expected_answer") or {}
    fails: list[str] = []
    resp_l = resp.lower()

    # must_contain_any_of: list of lists — each inner list needs >= 1 hit
    for group in ea.get("must_contain_any_of", []):
        if group and not any(t and t.lower() in resp_l for t in group):
            fails.append(f"missing any-of {group[:3]}")

    # must_not_contain: none of these may appear
    for bad in ea.get("must_not_contain", []):
        if bad and bad.lower() in resp_l:
            fails.append(f"contains forbidden: {bad}")

    # Phone extension / Employee ID leak prevention (refusal bucket)
    if ea.get("must_not_contain_phone_extension"):
        if re.search(r"\b\d{5}\b", resp):
            fails.append("leaked phone extension")
    if ea.get("must_not_contain_employee_id_pattern"):
        if re.search(r"\b0[08]\d{6}\b", resp):
            fails.append("leaked Employee ID")

    # min_items / exact_count / all_items_tokens_per_id
    tokens_per_id: dict = ea.get("all_items_tokens_per_id") or {}
    if tokens_per_id:
        matched_ids = []
        for emp_id, toks in tokens_per_id.items():
            if toks and any(t and t.lower() in resp_l for t in toks):
                matched_ids.append(emp_id)
        min_items = ea.get("min_items")
        if min_items is not None and len(matched_ids) < min_items:
            fails.append(f"min_items {len(matched_ids)}/{min_items}")
        exact_count = ea.get("exact_count")
        if exact_count is not None and len(matched_ids) != exact_count:
            fails.append(f"exact_count got {len(matched_ids)}, need {exact_count}")

    return (len(fails) == 0, fails)


def main():
    subs = load_submission(SUB_PATH)
    gt_items = load_ground_truth(GT_PATH)
    n = len(gt_items)

    by_bucket = Counter()
    by_bucket_pass = Counter()
    passed = 0
    missing = 0
    for gt in gt_items:
        iid = gt["id"]
        by_bucket[gt["bucket"]] += 1
        if iid not in subs:
            missing += 1
            continue
        ok, _ = grade_item(gt, subs[iid])
        if ok:
            passed += 1
            by_bucket_pass[gt["bucket"]] += 1

    print(f"Scored {n} items against {GT_PATH}")
    print(f"Passed: {passed}/{n} = {passed/n:.1%}")
    if missing:
        print(f"Missing from submission: {missing}")
    print()
    print(f"{'Bucket':32} {'pass/total':>12}  {'rate':>6}")
    print("-" * 56)
    for b in sorted(by_bucket, key=lambda k: -by_bucket[k]):
        p, t = by_bucket_pass[b], by_bucket[b]
        print(f"{b:32} {p}/{t:>8} {p/t*100:>6.1f}%")

    print()
    print(json.dumps({"score": passed / n, "passed": passed, "total": n}))


if __name__ == "__main__":
    main()
