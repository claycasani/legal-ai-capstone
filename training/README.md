# LoRA Fine-Tuning Plan

The project should keep RAG for factual grounding and use LoRA to teach an open
model the desired contract-explanation behavior.

## Goal

Fine-tune an instruct model to produce consumer-friendly contract explanations:

- plain-English summaries
- key obligations
- things to watch before signing
- questions to clarify
- evidence-grounded citations
- refusals when retrieved text does not support an answer

## Recommended Base Models

Start with one of these, depending on available Hugging Face Space or Colab GPU:

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `meta-llama/Llama-3.1-8B-Instruct`
- `microsoft/Phi-3.5-mini-instruct` for tighter hardware

## Dataset Format

Use JSONL with one example per line:

```json
{"instruction":"Explain the clause in plain English using only the retrieved context.","context":"[Chunk 1] ...","response":"## What It Says\n..."}
```

High-value example types:

- clause explanation
- whole-document summary from retrieved chunks
- Q&A over one or more retrieved chunks
- risky or one-sided term detection
- unanswerable questions where the correct response is to abstain

## Build The First Training Set

Generate a starter dataset from the public corpus:

```bash
python training/build_lora_dataset.py \
  --corpus-dir legal_corpus \
  --output training/generated_lora_examples.jsonl \
  --max-examples 240
```

Validate the JSONL before training:

```bash
python training/validate_lora_dataset.py training/generated_lora_examples.jsonl
```

Then combine the generated examples with the hand-written seed examples:

```bash
cat training/seed_lora_examples.jsonl training/generated_lora_examples.jsonl \
  > training/lora_train.jsonl
```

The generated dataset is a starting point, not the final gold dataset. The best
capstone version should manually review and improve examples for:

- rent/payment clauses with exact dollar amounts
- renewal and termination deadlines
- one-sided liability or indemnification
- unanswerable questions
- bad OCR/PDF extraction cases

## Training Command

```bash
python training/train_lora.py \
  --dataset training/lora_train.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --output-dir models/contract-clarity-lora
```

## Using the Adapter

After training, run the app with:

```bash
GENERATION_PROVIDER=lora \
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
LORA_ADAPTER_PATH=models/contract-clarity-lora \
python app.py
```

Embeddings still use OpenAI by default, so `OPENAI_API_KEY` is required unless
the retrieval layer is later switched to local sentence-transformer embeddings.
