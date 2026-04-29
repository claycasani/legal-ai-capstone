# Hugging Face Deployment

Deploy the app as a Hugging Face **Gradio Space** using the OpenAI backend.
This is the recommended capstone demo path because it runs on free CPU hardware.

## Why OpenAI Mode For Deployment?

The trained LoRA adapter proves domain adaptation, but running a local Qwen model
inside a Space requires paid GPU hardware. Hugging Face CPU Basic is free, but it
cannot run the LoRA model at usable speed.

Recommended demo:

```text
Hugging Face Space -> Gradio app -> OpenAI generation + RAG
```

Pitch/story:

```text
The public demo uses cost-efficient API inference for reliability. We also trained
a PEFT LoRA adapter as the domain-adaptation path for future GPU or dedicated
inference deployment.
```

## Create The Space

1. Go to Hugging Face.
2. Create a new Space.
3. SDK: `Gradio`.
4. Hardware: `CPU Basic`.
5. Visibility: public for demo, private/protected if using real documents.

Hugging Face docs: https://huggingface.co/docs/hub/spaces-overview

## Required Secret

In the Space settings, add a secret:

```text
OPENAI_API_KEY=<your key>
```

Do not hard-code the key in the repo.

Hugging Face secrets docs: https://huggingface.co/docs/hub/main/spaces-overview#managing-secrets-and-environment-variables

## Push Files

The Space needs:

```text
app.py
requirements.txt
src/
README.md
```

Do not upload:

```text
models/
training/*_lora_*.jsonl
training/lora_train.jsonl
training/lora_eval.jsonl
legal_corpus/ if it contains anything private
```

## Local Git Push Pattern

After creating the Space repo on Hugging Face:

```bash
git remote add hf-space https://huggingface.co/spaces/YOUR_USERNAME/contract-clarity
git push hf-space codex-lora-rag-ui:main
```

If you do not want to push the whole project repo, create a clean deployment
folder and copy only the files listed above.

## GPU LoRA Option

If you want to run the LoRA model live, upgrade the Space to GPU and set:

```text
GENERATION_PROVIDER=lora
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
LORA_ADAPTER_PATH=models/contract-clarity-lora
```

You would also need `requirements-lora.txt` packages installed and the adapter
available in the Space. This is not the recommended free deployment path.

Current Hugging Face GPU examples:

- T4 small: about $0.40/hour.
- L4: about $0.80/hour.
- A10G small: about $1.00/hour.

GPU billing runs while the Space is starting/running, so set sleep time or pause
the Space after demos.

GPU docs: https://huggingface.co/docs/hub/spaces-gpus
