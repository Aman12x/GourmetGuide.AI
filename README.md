# 🍽️ GourmetGuide AI: Multimodal RAG Food Recommender on Amazon Bedrock

GourmetGuide recommends dishes from a 50-item menu across 10 restaurants. You can ask in plain language ("something spicy and vegetarian under $10, no nuts") or upload a photo of a dish you like.

<img width="1440" height="802" alt="GourmetGuide screenshot" src="https://github.com/user-attachments/assets/61846b9c-8627-4b7a-8ceb-200dff5018d9" />

## How it works

```
photo ──► Claude (vision) describes the dish ─┐
                                              ▼
text + last 5 turns ──► Claude plan call (tool use) ──► intent, search query, hard filters
                                              │
                    chat ◄────────────────────┤
                                              ▼
              FAISS vector search (Titan Text Embeddings v2) over only the dishes
              that pass every filter: price, vegetarian, cuisine, allergens, rating
                                              │
                                              ▼
            Claude rank call (tool use) picks up to 3 candidates and writes a grounded summary
```

- **Bedrock Converse API** for every model call (`bedrock.py`). The model ID is configurable, and the default is Claude Haiku 5.5 through the US cross-Region inference profile.
- **Hybrid search** (`retrieval.py`):
  - Filters are hard constraints in code, applied over vector search, not instructions in a prompt.
  - Allergens are checked against both the listed dietary warnings and the ingredient list.
  - Allergen matching is deliberately broad, so a dish is excluded when in doubt.
- **Structured output via tool use.** The planner and the ranker each return schema-checked JSON.
  - Any dish ID the model invents or repeats is dropped.
  - If the model wraps JSON in prose, the parser still recovers it.
- **Calls per turn** (enforced in `tests/test_pipeline.py`): small talk 1, recommendation 2, plus 1 when a photo is attached. A filter that matches nothing is answered without a second call.
- **Bedrock Guardrails** (optional): set `BEDROCK_GUARDRAIL_ID` to apply a guardrail to every call. `scripts/create_guardrail.py` creates one with:
  - content filters
  - prompt-attack defence
  - PII blocking
  - a denied topic for medical or allergy-safety guarantees
- **Uploads**: the app detects the real image type (PNG, JPEG, GIF or WebP) and clears the photo after it has been used.

## Data and index

| File | Purpose |
|---|---|
| `data/dishes.jsonl` | Source of truth: one row per dish, with its image description and structured fields |
| `ingest.py` | Builds `output/faiss_index` from `dishes.jsonl` with Titan Text Embeddings v2. The embedded text includes the dish, restaurant and cuisine. |
| `data/images/` | Menu photos shown on the recommendation cards |
| `data/*.csv` | Original raw menu data, kept for reference |

The committed index was built before restaurant and cuisine were added to the embedded text. Run `python ingest.py` to rebuild it in the new format. Its metadata already supports every filter.

## Evaluation

`eval/cases.jsonl` holds 32 labelled requests. Each one carries the filters it should produce and the dishes that would satisfy it. `eval/run_eval.py` runs the live pipeline on Bedrock and reports:

- intent accuracy
- filter-extraction accuracy
- hit rate: a labelled dish is among the picks
- constraint violations: must be 0
- groundedness, judged per summary by a second Claude model

It exits non-zero when a gate fails and writes the full results to `eval/results/`.

```bash
python eval/run_eval.py            # needs AWS credentials with Bedrock access
```

The offline suite (`pytest`) fakes Bedrock and runs in CI on every push. It covers:
- filters and allergen matching
- call counts per turn
- request shapes, checked against botocore's Bedrock API model
- the Streamlit app itself, via `AppTest`
- the eval labels

## Setup

```bash
git clone https://github.com/Aman12x/GourmetGuide.AI.git
cd GourmetGuide.AI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Credentials come from the standard AWS chain (`aws configure`, SSO, an IAM role, or environment variables). On Streamlit Cloud, put them in `st.secrets`. Optional settings, as environment variables or secrets:

| Setting | Default |
|---|---|
| `AWS_DEFAULT_REGION` | `us-east-1` |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-haiku-5-5` |
| `BEDROCK_GUARDRAIL_ID` / `BEDROCK_GUARDRAIL_VERSION` | none / `DRAFT` |

Your account needs Bedrock model access to the Claude model and to `amazon.titan-embed-text-v2:0`.

```bash
python ingest.py          # rebuild the index after editing data/dishes.jsonl
streamlit run app.py
pip install -r requirements-dev.txt && pytest
```

## Project structure

```
├── app.py                 # Streamlit UI
├── pipeline.py            # One chat turn: describe image → plan → filtered search → rank
├── retrieval.py           # Metadata filters, allergen matching, filtered FAISS search
├── bedrock.py             # Converse wrapper: structured output, guardrails, image types
├── ingest.py              # Builds the FAISS index from data/dishes.jsonl
├── scripts/               # create_guardrail.py, export_dishes.py (one-time export)
├── eval/                  # Labelled cases and the live eval runner
├── tests/                 # Offline pytest suite (Bedrock faked)
└── output/faiss_index/    # Committed vector index
```

Developed by Aman Singh
