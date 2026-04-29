---
title: Contract Clarity
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
---

# Contract Clarity

Contract Clarity is a capstone MVP for helping everyday users understand legal
contracts. A user uploads a contract, the system retrieves relevant clauses, and
the model produces plain-English summaries, key-clause explanations, signer
risks, clarifying questions, and citation-backed Q&A.

The project is moving from notebook-only RAG toward a stronger capstone
architecture:

```text
contract upload -> parsing -> clause-aware chunks -> retrieval -> LoRA/RAG generation -> polished UI
```

## Current App

Run locally:

```bash
pip install -r requirements.txt
OPENAI_API_KEY=... python app.py
```

Then open the Gradio URL shown in the terminal.

## Generation Modes

Default mode uses OpenAI for generation:

```bash
GENERATION_PROVIDER=openai python app.py
```

After training a LoRA adapter, switch generation to the local fine-tuned model:

```bash
GENERATION_PROVIDER=lora \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
LORA_ADAPTER_PATH=models/contract-clarity-lora \
python app.py
```

Embeddings currently use OpenAI `text-embedding-3-small` in both modes.

## Project Layout

- `app.py` - deployable Gradio app for upload, review, key clauses, and Q&A.
- `src/legal_ai/parsing.py` - PDF, DOCX, and HTML text extraction.
- `src/legal_ai/chunking.py` - clause-aware chunking and metadata tagging.
- `src/legal_ai/retrieval.py` - temporary Chroma collections for uploaded docs.
- `src/legal_ai/generation.py` - OpenAI generation now, LoRA generation later.
- `training/` - LoRA dataset format, seed examples, and training script.
- `evaluation/` - fixed regression prompts from the capstone evaluation plan.
- `notebooks/` - original milestone notebooks and saved retrieval artifacts.

## Hugging Face Spaces

For a Gradio Space, commit `app.py`, `requirements.txt`, `src/`, `training/`,
and `evaluation/`. Add `OPENAI_API_KEY` as a Space secret.

GPU deployment is only needed when running `GENERATION_PROVIDER=lora` with a
local adapter. The default OpenAI mode can run on CPU.

For a live LoRA demo Space, upload `models/contract-clarity-lora` and set:

```text
GENERATION_PROVIDER=lora
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
LORA_ADAPTER_PATH=models/contract-clarity-lora
OPENAI_API_KEY=<Space secret>
```

After the demo, pause the Space or switch the hardware back to CPU Basic to stop
GPU billing.
