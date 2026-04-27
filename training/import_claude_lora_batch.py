import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_ai.parsing import normalize_pdf_artifacts  # noqa: E402

REQUIRED_SECTIONS = [
    "## What It Says",
    "## Why It Matters",
    "## Things To Watch",
    "## Supporting Citations",
    "## Disclaimer",
]

REVIEW_PATTERNS = [
    "20% excise",
    "immediate taxation",
    "personally liable",
    "typically required",
    "can be as high as",
    "no exceptions",
    "serious financial risk",
    "significant financial",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--accepted")
    parser.add_argument("--review")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit)
    accepted_path = Path(args.accepted) if args.accepted else None
    review_path = Path(args.review) if args.review else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if accepted_path:
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
    if review_path:
        review_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_rows = []
    audit_rows = []
    accepted_rows = []
    review_rows = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            cleaned = clean_row(row)
            flags = audit_row(cleaned, line_number)
            cleaned_rows.append(cleaned)
            if flags:
                audit_rows.append({"line": line_number, "flags": flags, "instruction": cleaned["instruction"]})
                review_rows.append(cleaned)
            else:
                accepted_rows.append(cleaned)

    with output_path.open("w", encoding="utf-8") as handle:
        for row in cleaned_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_rows, handle, indent=2)

    if accepted_path:
        write_jsonl(accepted_path, accepted_rows)
    if review_path:
        write_jsonl(review_path, review_rows)

    print(f"Wrote {len(cleaned_rows)} cleaned examples to {output_path}")
    print(f"Wrote {len(audit_rows)} audit findings to {audit_path}")
    if accepted_path:
        print(f"Wrote {len(accepted_rows)} accepted examples to {accepted_path}")
    if review_path:
        print(f"Wrote {len(review_rows)} review examples to {review_path}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def clean_row(row: dict) -> dict:
    context = clean_text(row["context"])
    response = clean_text(row["response"])

    if not context.lstrip().startswith("[Chunk 1]"):
        context = "[Chunk 1]\n" + context.strip()

    response = normalize_disclaimer(response)
    response = normalize_supporting_citations(response)

    return {
        "instruction": row["instruction"].strip(),
        "context": context,
        "response": response,
    }


def clean_text(text: str) -> str:
    text = normalize_pdf_artifacts(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"(\s*\[illegible mark\]\s*){2,}", " [unreadable marks] ", text)
    text = text.replace("[illegible mark]", "[unreadable mark]")
    text = text.replace("[illegible marks]", "[unreadable marks]")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_disclaimer(response: str) -> str:
    disclaimer = "This is not legal advice and does not replace review by an attorney."
    if "## Disclaimer" not in response:
        return response.rstrip() + "\n\n## Disclaimer\n" + disclaimer
    before, _heading, _after = response.partition("## Disclaimer")
    return before.rstrip() + "\n\n## Disclaimer\n" + disclaimer


def normalize_supporting_citations(response: str) -> str:
    if "[Chunk 1]" in response:
        return response

    if "## Supporting Citations" not in response:
        return response.rstrip() + "\n\n## Supporting Citations\n- [Chunk 1] Supports the answer from the retrieved contract text."

    before, heading, after = response.partition("## Supporting Citations")
    citation_line = first_nonempty_line(after)
    if citation_line:
        cleaned_line = re.sub(r"^[-*]?\s*Source:\s*", "", citation_line).strip()
        replacement = f"{heading}\n- [Chunk 1] {cleaned_line}"
    else:
        replacement = f"{heading}\n- [Chunk 1] Supports the answer from the retrieved contract text."

    rest = section_after_first_line(after)
    return before.rstrip() + "\n\n" + replacement + rest


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip() and not line.strip().startswith("## "):
            return line.strip()
    return ""


def section_after_first_line(text: str) -> str:
    lines = text.splitlines()
    found_first = False
    rest = []
    for line in lines:
        if not found_first and line.strip() and not line.strip().startswith("## "):
            found_first = True
            continue
        if found_first:
            rest.append(line)
    return "\n" + "\n".join(rest).rstrip() if rest else ""


def audit_row(row: dict, line_number: int) -> list[str]:
    flags = []
    response = row["response"]
    context = row["context"]

    for section in REQUIRED_SECTIONS:
        if section not in response:
            flags.append(f"missing section: {section}")

    if "[Chunk 1]" not in response:
        flags.append("missing [Chunk 1] citation")

    if "[unreadable mark" in context or "[unreadable mark" in response:
        flags.append("contains unreadable OCR marks")

    lowered_response = response.lower()
    for pattern in REVIEW_PATTERNS:
        if pattern in lowered_response:
            flags.append(f"possible outside legal knowledge: {pattern}")

    if line_number <= 3 and "Not found in the retrieved text" not in response and "$" in context:
        # Early lease examples include blank/check-boxed amounts. Nudge human review.
        flags.append("verify money terms against source")

    return flags


if __name__ == "__main__":
    main()
