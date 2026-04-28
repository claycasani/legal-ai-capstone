# VC Cost Model

Pricing snapshot: April 28, 2026.

## Current RAG MVP

The app currently uses:

- OpenAI `text-embedding-3-small` for document chunk embeddings.
- OpenAI `gpt-4.1-mini` for answer generation.
- Chroma for local vector search.
- Gradio UI, deployable to Hugging Face Spaces.

OpenAI published pricing:

- `text-embedding-3-small`: $0.02 per 1M tokens.
- `gpt-4.1-mini`: $0.40 per 1M input tokens, $1.60 per 1M output tokens.

Example per-document estimate:

| Stage | Assumption | Estimated Cost |
| --- | ---: | ---: |
| Embed upload | 20,000 document tokens | $0.0004 |
| Generate summary | 10,000 input + 1,500 output tokens | $0.0064 |
| Key clauses | 25,000 input + 4,000 output tokens | $0.0164 |
| One Q&A | 5,000 input + 800 output tokens | $0.0033 |

Approximate cost for one user reviewing one contract with a summary, clause
watchlist, and three questions:

```text
$0.0004 embeddings
$0.0064 summary
$0.0164 key clauses
3 x $0.0033 Q&A
= about $0.033 per reviewed document
```

Pitch shorthand:

```text
The current API-backed MVP can support consumer contract reviews at roughly
3-5 cents per full document session, depending on document length and number of
follow-up questions.
```

## LoRA Path

LoRA does not automatically reduce all costs. It changes the cost structure.

### Training Cost

Training is a one-time or occasional GPU cost. Hugging Face Spaces GPU examples:

- Nvidia T4 small: $0.40/hour.
- Nvidia L4: $0.80/hour.
- Nvidia A10G small: $1.00/hour.
- Nvidia A100 large: $2.50/hour.

If QLoRA training takes 2-6 hours:

```text
T4: about $0.80-$2.40
L4: about $1.60-$4.80
A10G: about $2.00-$6.00
A100: about $5.00-$15.00
```

### Inference Cost

Running the LoRA model live requires GPU hosting unless we use a separate
inference provider. GPU Spaces are billed while running, so a GPU demo should
use sleep/pausing when not in use.

For the pitch, position LoRA as:

- better domain behavior
- better structured explanations
- reduced dependency on proprietary generation over time
- potential margin improvement at scale if inference is optimized

Do not pitch LoRA as automatically cheaper for the prototype. For low traffic,
OpenAI API calls are cheaper and simpler. For higher traffic, a hosted
fine-tuned open model may become attractive if utilization is high enough.

## VC-Style Unit Economics

Possible consumer pricing:

| Plan | Price | Usage | Gross Cost Estimate |
| --- | ---: | --- | ---: |
| One-off review | $4.99 | 1 contract + Q&A | $0.03-$0.10 |
| Student/basic | $9.99/mo | 5 contracts | $0.15-$0.50 |
| Power user | $19.99/mo | 20 contracts | $0.60-$2.00 |

Core pitch:

```text
Contract Clarity turns a dense legal document into a consumer-facing review for
pennies in variable AI cost, leaving room for strong gross margins even at
low-price consumer tiers.
```

## Cost Risks

- Long documents increase context and generation tokens.
- Repeated Q&A can dominate cost if users chat heavily.
- GPU LoRA inference is wasteful if the demo runs 24/7 with low traffic.
- Bad retrieval increases cost by passing irrelevant chunks to the model.

## Cost Controls

- Cap retrieved chunks per response.
- Cache document embeddings per upload/session.
- Cache generated summary and clause watchlist.
- Use API generation for low traffic; use LoRA/GPU only for demos or scale tests.
- Put paid GPU Spaces to sleep when idle.
