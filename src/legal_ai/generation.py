import json
import os
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from legal_ai.config import (
    CLAUSE_QUERIES,
    DEFAULT_LOCAL_GENERATION_MODEL,
    DEFAULT_LORA_ADAPTER_PATH,
    OPENAI_CHAT_MODEL,
    SUMMARY_QUERIES,
)
from legal_ai.retrieval import ContractRetriever, build_context

QA_SYSTEM_PROMPT = """You are a legal document assistant for everyday consumers.

Rules:
- Use ONLY the retrieved contract context.
- Do NOT provide legal advice.
- Do NOT invent facts, clauses, or interpretations.
- If the context does not support an answer, say so clearly.
- Explain legal terms in plain English.
- Always include supporting citations to retrieved chunks.
- Keep paragraphs to 1-2 short sentences.
- Use at most 4 bullets per section.
- State exact amounts, dates, deadlines, party names, notice periods, and required actions when the context contains them.
- Do not say what a clause "explains" or "covers"; say what the clause actually requires, allows, or prohibits.

Return concise Markdown with these headings:
Plain-English Answer
Key Risks / Important Notes
Supporting Citations
Disclaimer
"""

SUMMARY_SYSTEM_PROMPT = """You are a legal document assistant for everyday consumers.

Rules:
- Summarize ONLY the provided contract context.
- Do NOT provide legal advice.
- Do NOT invent facts or obligations not supported by the text.
- Explain legal terms in plain English.
- Always include supporting citations to retrieved chunks.
- Keep paragraphs to 1-2 short sentences.
- Use at most 4 bullets per section.
- Write for someone deciding what to ask before signing.
- State exact amounts, dates, deadlines, party names, notice periods, and required actions when the context contains them.
- If the contract is a lease, the summary must include rent amount, rent due date, deposits, late fees, and utilities when retrieved.
- Do not say what a clause "explains" or "covers"; say what the clause actually requires, allows, or prohibits.

Return concise Markdown with these headings:
Plain-English Summary
Money Snapshot
Key Obligations / Important Terms
Things To Watch
Questions To Clarify
Supporting Citations
Disclaimer
"""

CLAUSE_SYSTEM_PROMPT = """You are a legal document assistant for everyday consumers.

Rules:
- Explain ONLY the retrieved clause text.
- Do NOT provide legal advice.
- Do NOT invent terms that are not present in the text.
- Explain legal terms in plain English.
- Always include supporting citations to retrieved chunks.
- Keep each section short.
- Use at most 3 bullets per section.
- Make the first sentence explain the clause in everyday language.
- State exact amounts, dates, deadlines, party names, notice periods, and required actions when the context contains them.
- Do not say "this clause explains..." or "this clause outlines..."; say the actual term, for example "Rent is $2,100 per month and is due on the 1st."
- If a value is not present in the retrieved text, say "Not found in the retrieved text."

Return concise Markdown with these headings:
What It Says
Why It Matters
Things To Watch
Supporting Citations
Disclaimer
"""


class Generator:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIGenerator(Generator):
    def __init__(self, client: OpenAI, model: str = OPENAI_CHAT_MODEL):
        self.client = client
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


class LocalLoraGenerator(Generator):
    def __init__(
        self,
        base_model: str = DEFAULT_LOCAL_GENERATION_MODEL,
        adapter_path: str = DEFAULT_LORA_ADAPTER_PATH,
    ):
        adapter_path = str(adapter_path)
        if not Path(adapter_path).exists():
            raise FileNotFoundError(
                "LoRA generation is enabled, but the adapter folder was not found at "
                f"{adapter_path}. Upload models/contract-clarity-lora to the Space or "
                "set GENERATION_PROVIDER=openai."
            )

        try:
            from peft import PeftModel
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "LoRA generation requires peft, torch, transformers, accelerate, and "
                "bitsandbytes. Install requirements.txt before starting the app."
            ) from exc

        base_model = resolve_lora_base_model(adapter_path, base_model)

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            # Use float16 — T4/A10G support it; bfloat16 is A100/H100 only
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            model_kwargs["torch_dtype"] = "auto"

        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=int(os.getenv("LORA_MAX_NEW_TOKENS", "700")),
            temperature=0.2,
            do_sample=False,
            return_full_text=False,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = (
            "<|system|>\n"
            f"{system_prompt}\n"
            "<|user|>\n"
            f"{user_prompt}\n"
            "<|assistant|>\n"
        )
        output = self.pipe(prompt)[0]["generated_text"]
        return output.split("<|assistant|>")[-1].strip()


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for embeddings and OpenAI generation.")
    return OpenAI(api_key=api_key)


def get_generator() -> Generator:
    provider = os.getenv("GENERATION_PROVIDER")
    if provider is None:
        provider = "lora" if Path(os.getenv("LORA_ADAPTER_PATH", DEFAULT_LORA_ADAPTER_PATH)).exists() else "openai"
    provider = provider.lower()
    if provider == "lora":
        return LocalLoraGenerator(
            base_model=os.getenv("BASE_MODEL", DEFAULT_LOCAL_GENERATION_MODEL),
            adapter_path=os.getenv("LORA_ADAPTER_PATH", DEFAULT_LORA_ADAPTER_PATH),
        )
    return OpenAIGenerator(get_openai_client())


def resolve_lora_base_model(adapter_path: str, fallback: str) -> str:
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        return fallback
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback
    return config.get("base_model_name_or_path") or fallback


class ContractAssistant:
    def __init__(self, retriever: ContractRetriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def answer_question(self, question: str, collection_name: str, source_document: str) -> str:
        if not question.strip():
            return "Please enter a question."
        rows = self.retriever.retrieve(question, collection_name, top_k=6, initial_k=20)
        context = build_context(rows)
        if context == "NO_RELEVANT_CONTEXT_FOUND":
            return missing_context_response("Plain-English Answer")
        prompt = f"Uploaded Document: {source_document}\n\nUser Question:\n{question}\n\nRetrieved Context:\n{context}"
        return with_source_text(self.generator.generate(QA_SYSTEM_PROMPT, prompt), rows)

    def summarize(self, collection_name: str, source_document: str) -> str:
        rows = self._collect_unique_rows(SUMMARY_QUERIES, collection_name, top_k=4, initial_k=18)
        rows = merge_rows(rows, self.payment_rows(collection_name), limit=16)
        context = build_context(rows)
        if context == "NO_RELEVANT_CONTEXT_FOUND":
            return missing_context_response("Plain-English Summary")
        prompt = (
            f"Uploaded Document: {source_document}\n\n"
            "Task: Create a consumer-friendly contract review using only the context. "
            "Extract concrete terms, especially money, dates, deadlines, required actions, and warning signs.\n\n"
            f"Retrieved Context:\n{context}"
        )
        return with_source_text(self.generator.generate(SUMMARY_SYSTEM_PROMPT, prompt), rows)

    def key_clauses(self, collection_name: str, source_document: str) -> str:
        sections = []
        for clause_name, query in CLAUSE_QUERIES.items():
            rows = self.retriever.retrieve(query, collection_name, top_k=5, initial_k=18)
            if clause_name == "Payment":
                rows = merge_rows(self.payment_rows(collection_name), rows, limit=6)
            context = build_context(rows)
            if context == "NO_RELEVANT_CONTEXT_FOUND":
                continue
            prompt = (
                f"Uploaded Document: {source_document}\n\n"
                f"Clause Category: {clause_name}\n\n"
                "Task: Extract what this clause category actually says using only the context. "
                "Use concrete terms and cite chunks. Do not describe the clause generically.\n\n"
                f"Retrieved Context:\n{context}"
            )
            generated = self.generator.generate(CLAUSE_SYSTEM_PROMPT, prompt)
            sections.append(f"## {clause_name}\n\n{with_source_text(generated, rows)}")
        return "\n\n".join(sections) if sections else missing_context_response("Clause Summary")

    def payment_rows(self, collection_name: str) -> list[dict]:
        return self.retriever.retrieve_keyword(
            collection_name,
            keywords=[
                "rent",
                "monthly rent",
                "base rent",
                "payment",
                "due",
                "deposit",
                "late charge",
                "late fee",
                "returned check",
                "utility",
                "utilities",
                "fee",
                "charge",
            ],
            top_k=5,
            require_money=True,
        )

    def _collect_unique_rows(self, queries: list[str], collection_name: str, top_k: int, initial_k: int) -> list[dict]:
        rows = []
        seen_ids = set()
        for query in queries:
            for row in self.retriever.retrieve(query, collection_name, top_k=top_k, initial_k=initial_k):
                if row["id"] not in seen_ids:
                    rows.append(row)
                    seen_ids.add(row["id"])
        return rows


def missing_context_response(heading: str) -> str:
    return (
        f"## {heading}\n\n"
        "I could not find enough supporting text in the uploaded document to answer reliably.\n\n"
        "## Supporting Citations\n\n"
        "- None.\n\n"
        "## Disclaimer\n\n"
        "This is not legal advice and does not replace review by an attorney."
    )


def with_source_text(markdown_text: str, rows: list[dict]) -> str:
    if not rows:
        return markdown_text

    lines = ["\n\n## Source Text"]
    for index, row in enumerate(rows[:4], start=1):
        meta = row.get("metadata", {})
        excerpt = compact_excerpt(row.get("document", ""))
        chunk_index = meta.get("chunk_index", "?")
        lines.append(
            f"- [Chunk {index}](#chunk-{index}) | source chunk {chunk_index}: {excerpt}"
        )
    return markdown_text.rstrip() + "\n" + "\n".join(lines)


def compact_excerpt(text: str, limit: int = 260) -> str:
    compact = " ".join(text.split())
    compact = compact.replace("[illegible marks]", "[unreadable marks]")
    compact = compact.replace("[illegible mark]", "[unreadable mark]")
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def merge_rows(primary: list[dict], secondary: list[dict], limit: int) -> list[dict]:
    rows = []
    seen = set()
    for row in primary + secondary:
        row_id = row.get("id")
        if row_id in seen:
            continue
        rows.append(row)
        seen.add(row_id)
        if len(rows) >= limit:
            break
    return rows
