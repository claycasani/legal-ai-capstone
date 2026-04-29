# Hugging Face Deployment

Deploy the app as a Hugging Face **Gradio Space**. The app can run in either
OpenAI mode on free CPU hardware or LoRA mode on paid GPU hardware.

## Deployment Modes

OpenAI mode is the lowest-friction public demo path because it runs on free CPU
hardware.

```text
Hugging Face Space -> Gradio app -> OpenAI generation + RAG
```

LoRA mode is the strongest capstone demo path because it runs the trained PEFT
adapter, but it requires GPU hardware because the adapter still loads a base
model.

```text
Hugging Face Space GPU -> Gradio app -> Qwen 3B + LoRA adapter + RAG
```

Pitch/story:

```text
The free public demo can run with API inference for reliability. The investor
demo can temporarily switch to GPU-backed LoRA inference, then pause the Space
when the live demo is over.
```

## Create The Space

1. Go to Hugging Face.
2. Create a new Space.
3. SDK: `Gradio`.
4. Hardware: `CPU Basic` for OpenAI mode, or `T4 small` / `L4` for LoRA mode.
5. Visibility: public for demo, private/protected if using real documents.

Hugging Face docs: https://huggingface.co/docs/hub/spaces-overview

## Required Secrets And Variables

In the Space settings, add this secret:

```text
OPENAI_API_KEY=<your key>
```

Do not hard-code the key in the repo.

For LoRA mode, also add these variables:

```text
GENERATION_PROVIDER=lora
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
LORA_ADAPTER_PATH=models/contract-clarity-lora
```

For CPU/OpenAI mode, use:

```text
GENERATION_PROVIDER=openai
```

Hugging Face secrets docs: https://huggingface.co/docs/hub/main/spaces-overview#managing-secrets-and-environment-variables

## Push Files

The safest LoRA deploy path is to build a clean Space bundle:

```bash
python3 scripts/build_hf_space_bundle.py
```

This creates:

```text
dist/hf_space/
```

Upload or push the contents of that folder to your Hugging Face Space. The
bundle includes only the files needed for inference and skips the large
`checkpoint-*` training folders.

If you copy files manually, the Space needs:

```text
app.py
requirements.txt
src/
README.md
```

For LoRA mode, also upload:

```text
models/contract-clarity-lora/
```

Do not upload private training data or private source contracts:

```text
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

The main `requirements.txt` already includes the LoRA runtime packages, so the
same repo can run on a GPU Space. The adapter must be available in the Space at
`models/contract-clarity-lora`.

Current Hugging Face GPU examples:

- T4 small: about $0.40/hour.
- L4: about $0.80/hour.
- A10G small: about $1.00/hour.

GPU billing runs while the Space is starting/running. Pause the Space after
testing or demos. You can also set a custom sleep time in Space settings.

GPU docs: https://huggingface.co/docs/hub/spaces-gpus
