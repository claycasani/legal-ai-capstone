import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_ai.chunking import chunk_document  # noqa: E402
from legal_ai.parsing import parse_document  # noqa: E402

CLAUSE_PRIORITY = [
    ("Payment", "Payment / financial terms"),
    ("Termination", "Termination / cancellation"),
    ("Liability", "Liability / indemnification"),
    ("Confidentiality", "Confidentiality / non-disclosure"),
    ("IP Ownership", "IP ownership / licensing"),
    ("Governing Law", "Governing law / jurisdiction"),
    ("Dispute Resolution", "Dispute resolution / arbitration"),
    ("Warranties", "Warranties / representations"),
]

QUERY_BY_CLAUSE = {
    "Payment": "What does the signer have to pay, how much, and when?",
    "Termination": "How can this agreement end, and what notice is required?",
    "Liability": "What liability, indemnity, or damage obligations does this create?",
    "Confidentiality": "What confidential information duties does this create?",
    "IP Ownership": "Who owns or can use intellectual property or work product?",
    "Governing Law": "What law, court, or venue controls disputes?",
    "Dispute Resolution": "How are disputes handled?",
    "Warranties": "What promises, disclaimers, or warranty limits does this include?",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="legal_corpus")
    parser.add_argument("--output", default="training/generated_lora_examples.jsonl")
    parser.add_argument("--max-examples", type=int, default=240)
    args = parser.parse_args()

    examples = []
    for file_path in sorted(Path(args.corpus_dir).glob("*")):
        if file_path.suffix.lower() not in {".pdf", ".docx", ".html", ".htm"}:
            continue
        try:
            text = parse_document(file_path)
        except Exception as exc:
            print(f"Skipping {file_path.name}: {exc}", file=sys.stderr)
            continue

        for chunk in chunk_document(text):
            if is_low_quality_chunk(chunk["chunk_text"]):
                continue
            clause_name = choose_clause(chunk["clause_tags"], chunk["chunk_text"])
            if not clause_name:
                continue
            context = build_context(file_path.name, chunk)
            examples.append(build_clause_example(clause_name, context, chunk["chunk_text"]))
            examples.append(build_qa_example(clause_name, context, chunk["chunk_text"]))
            if len(examples) >= args.max_examples:
                break
        if len(examples) >= args.max_examples:
            break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples[: args.max_examples]:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")

    print(f"Wrote {min(len(examples), args.max_examples)} examples to {output_path}")


def choose_clause(tags: list[str], text: str) -> str:
    for clause_name, tag in CLAUSE_PRIORITY:
        if tag in tags:
            if clause_name == "Payment" and not is_payment_chunk(text):
                continue
            return clause_name
    return ""


def is_payment_chunk(text: str) -> bool:
    lowered = text.lower()
    has_money = bool(re.search(r"\$\s?\d[\d,]*(?:\.\d{2})?", text))
    payment_terms = [
        "rent",
        "monthly rent",
        "payment",
        "deposit",
        "late fee",
        "late charge",
        "returned check",
        "invoice",
        "compensation",
        "salary",
        "subscription fee",
        "fees are",
        "fee shall",
        "charges",
        "utilities",
    ]
    title_only_terms = ["fee simple", "fee title", "attorney's fees", "attorneys' fees"]
    if any(term in lowered for term in title_only_terms) and not has_money:
        return False
    return has_money or any(term in lowered for term in payment_terms)


def is_low_quality_chunk(text: str) -> bool:
    if len(text.split()) < 80:
        return True
    if text.count("_") > 20:
        return True
    if text.count("~") + text.count("^") + text.count("|") > 18:
        return True
    readable_chars = sum(char.isalnum() or char.isspace() or char in ".,;:$%()/-'" for char in text)
    if readable_chars / max(len(text), 1) < 0.88:
        return True
    return False


def build_context(source_document: str, chunk: dict) -> str:
    return (
        "[Chunk 1]\n"
        f"Source Document: {source_document}\n"
        f"Chunk Index: {chunk['chunk_index']}\n"
        f"Clause Tags: {' | '.join(chunk['clause_tags'])}\n"
        f"Text:\n{trim(chunk['chunk_text'], 2200)}"
    )


def build_clause_example(clause_name: str, context: str, chunk_text: str) -> dict:
    return {
        "instruction": (
            f"Extract what the {clause_name} clause actually says in plain English. "
            "Use concrete amounts, dates, deadlines, parties, and required actions when present. "
            "Do not describe the clause generically."
        ),
        "context": context,
        "response": build_response(clause_name, chunk_text),
    }


def build_qa_example(clause_name: str, context: str, chunk_text: str) -> dict:
    return {
        "instruction": (
            f"Question: {QUERY_BY_CLAUSE[clause_name]} Answer using only the retrieved context."
        ),
        "context": context,
        "response": build_response(clause_name, chunk_text),
    }


def build_response(clause_name: str, chunk_text: str) -> str:
    concrete_terms = extract_concrete_terms(chunk_text)
    key_sentences = extract_key_sentences(clause_name, chunk_text)

    what_it_says = key_sentences[:3]
    if concrete_terms:
        what_it_says.append("Concrete terms found: " + "; ".join(concrete_terms[:8]) + ".")
    if not what_it_says:
        what_it_says.append("The retrieved text does not provide enough clean detail to extract a concrete term.")

    return (
        "## What It Says\n"
        + bullets(what_it_says)
        + "\n\n## Why It Matters\n"
        + bullets(why_it_matters(clause_name))
        + "\n\n## Things To Watch\n"
        + bullets(things_to_watch(clause_name, concrete_terms))
        + "\n\n## Supporting Citations\n"
        + "- [Chunk 1] Supports the extracted terms because the statements above come from the retrieved text.\n\n"
        + "## Disclaimer\n"
        + "This is not legal advice and does not replace review by an attorney."
    )


def extract_concrete_terms(text: str) -> list[str]:
    patterns = [
        r"\$\s?\d[\d,]*(?:\.\d{2})?",
        r"\b\d+(?:\.\d+)?\s?%",
        r"\b\d+\s+(?:day|days|month|months|year|years)\b",
        r"\b(?:first|second|third|fourth|fifth|last)\s+day\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b(?:rent|payment|fee|deposit|charge|notice|renewal|termination|utilities?)\b[^.]{0,120}",
    ]
    terms = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            term = " ".join(match.split())
            if term and term not in terms:
                terms.append(term)
    return terms


def extract_key_sentences(clause_name: str, text: str) -> list[str]:
    keywords = keyword_set(clause_name)
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    matches = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            matches.append(trim(sentence, 280))
        if len(matches) >= 3:
            break
    return matches


def keyword_set(clause_name: str) -> list[str]:
    return {
        "Payment": ["rent", "payment", "fee", "deposit", "charge", "due", "$", "utility"],
        "Termination": ["terminate", "termination", "cancel", "renewal", "notice", "default"],
        "Liability": ["liability", "indemnify", "damage", "loss", "responsible"],
        "Confidentiality": ["confidential", "non-disclosure", "proprietary"],
        "IP Ownership": ["intellectual property", "work product", "license", "copyright"],
        "Governing Law": ["governing law", "jurisdiction", "venue", "court"],
        "Dispute Resolution": ["dispute", "arbitration", "litigation", "court"],
        "Warranties": ["warranty", "represent", "disclaimer", "as is"],
    }.get(clause_name, [])


def why_it_matters(clause_name: str) -> list[str]:
    return {
        "Payment": ["This affects the signer's actual cost and payment timing."],
        "Termination": ["This affects how easily the signer can exit or avoid renewal."],
        "Liability": ["This can shift financial responsibility for losses or claims."],
        "Confidentiality": ["This can create duties that continue after the relationship ends."],
        "IP Ownership": ["This affects who can own, reuse, or control created work."],
        "Governing Law": ["This affects where and under what law disputes may be handled."],
        "Dispute Resolution": ["This affects whether disputes go to court, arbitration, or another process."],
        "Warranties": ["This affects what promises are made and what protections may be limited."],
    }.get(clause_name, ["This term may affect the signer's rights or obligations."])


def things_to_watch(clause_name: str, concrete_terms: list[str]) -> list[str]:
    watch = {
        "Payment": ["Confirm every dollar amount, due date, late fee, deposit, and recurring charge."],
        "Termination": ["Check notice deadlines, automatic renewal, penalties, and duties after termination."],
        "Liability": ["Check whether responsibility is one-sided or unlimited."],
        "Confidentiality": ["Check how long the duty lasts and what information is covered."],
        "IP Ownership": ["Check whether ownership transfers or only a license is granted."],
        "Governing Law": ["Check whether the selected forum is inconvenient or costly."],
        "Dispute Resolution": ["Check whether jury trial, court access, or class claims are waived."],
        "Warranties": ["Check whether important promises are disclaimed or capped."],
    }.get(clause_name, ["Ask what this term requires in practical terms."])
    if not concrete_terms:
        watch.append("No exact amount, date, or deadline was cleanly extracted from this chunk.")
    return watch


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def trim(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


if __name__ == "__main__":
    main()
