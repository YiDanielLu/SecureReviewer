"""API-model inference for SecureReviewer (parity runner for the Harbor adapter).

Upstream's infer/test.py only runs the fine-tuned local models. This script does the same
job for an API model: it reviews each selected item of dataset/test.jsonl in one Messages API
call, parses the reply with upstream's `parse_security_review` (unchanged), and writes
result.jsonl in the format evaluate/SecureBleu.py reads. The Harbor adapter's parity agent
sends the same prompt with the same settings (single_call.py is shared byte-for-byte).

Prompt: upstream's review prompt (prompt1 in infer/test.py), the raw unified diff, and the
output-format section the Harbor task instruction adds so a general chat model produces the
"Security type: / Description: / Impact: / Advice:" layout the parser expects (the fine-tuned
models learned it from training data).

Usage (from the repo root):
    python infer/test_api.py --model claude-haiku-4-5 --indices 0 12 ... --out runs/run1
    cd runs/run1 && python ../../evaluate/SecureBleu.py
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_call import MAX_TOKENS, TEMPERATURE, call_claude, response_text  # noqa: E402

PROMPT = """You are a highly capable code reviewer specializing in security assessments. Your primary task is to conduct a comprehensive security review of the provided code changes. Identify and evaluate any potential security weaknesses, and generate a detailed review report that includes the following sections:
1. Security Type //The vulnerability type detected
2. Description //Clearly explain the security issue found in the provided code patch.
3. Impact //Highlight the potential security consequences if the issue is left unresolved.
4. Advice //Offer recommendations for resolving the issue.
If you judge that there is no security risk, output No Issue.

The Security Type must be exactly one of: Input Validation, Exception Handling, Type and Data Handling, Concurrency, Access Control and Information Security, Error State Management, Resource Management; or No Issue if no security vulnerability is detected.

### Code Change:

The code change (a unified diff hunk) is shown below and is also available at `/workspace/patch.diff`.

{diff_block}

### Output

Write your review comment to `/workspace/review.txt` in exactly this format (each header on its own line, followed by its content):

```
Security type:
<one security type from the list above>
Description:
<description of the security issue>
Impact:
<potential security consequences>
Advice:
<recommendations for resolving the issue>
```
"""


def render_diff_block(patch: str) -> str:
    fence = "~~~~"
    while fence in patch:
        fence += "~"
    return f"{fence}diff\n{patch.rstrip()}\n{fence}"


def build_prompt(patch: str) -> str:
    return PROMPT.replace("{diff_block}", render_diff_block(patch))


def parse_security_review(report):  # verbatim from infer/test.py
    import re

    sections = ["Security type:", "Description:", "Impact:", "Advice:"]
    pattern = r"(" + "|".join(sections) + r")\s*(.*?)\s*(?=(?:" + "|".join(sections) + r")|$)"
    matches = re.findall(pattern, report, re.DOTALL)

    review = {}
    for match in matches:
        key = match[0].rstrip(":")
        value = match[1].strip()
        review[key] = value
    for section in sections:
        key = section.rstrip(":")
        if key not in review:
            review[key] = None

    return review


def review_one(idx: int, sample: dict, model: str, out: Path) -> dict:
    raw_path = out / "raw" / f"test-{idx:03d}.json"
    if raw_path.exists():
        response = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        response = call_claude(build_prompt(sample["patch"]), model)
        raw_path.write_text(json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")
    parsed = parse_security_review(response_text(response))
    # Same field mapping as validate_and_save in infer/test.py. Missing sections become ""
    # rather than None, since SecureBleu.py fails on None (the Harbor grader does the same).
    row = dict(sample)
    row["Security Type"] = parsed["Security type"] if parsed["Security type"] is not None else "No Issue"
    row["Description"] = parsed["Description"] or ""
    row["Impact"] = parsed["Impact"] or ""
    row["Advice"] = parsed["Advice"] or ""
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--indices", type=int, nargs="+", required=True, help="0-based line numbers in dataset/test.jsonl")
    ap.add_argument("--test-file", default=str(Path(__file__).resolve().parents[1] / "dataset" / "test.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    items = [json.loads(line) for line in Path(args.test_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({"model": args.model, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "indices": args.indices}, indent=2),
        encoding="utf-8",
    )

    with ThreadPoolExecutor(args.parallel) as pool:
        rows = list(pool.map(lambda i: review_one(i, items[i], args.model, out), args.indices))

    with (out / "result.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} reviews to {out / 'result.jsonl'}")


if __name__ == "__main__":
    main()
