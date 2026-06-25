# Enterprise RAG Agent

> **Production-grade RAG pipeline with multi-agent routing, cross-encoder reranking, and LangSmith observability** — the exact AI stack enterprises need but can't deploy without senior AI talent.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.3+-009688?logo=chainlink&logoColor=white)](https://langchain.com)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-F97316)](https://smith.langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-F59E0B)](LICENSE)

---

## What This Project Demonstrates

This is **not a tutorial chatbot**. It is a production AI system built the way a Forward Deployed Engineer would deploy it inside an enterprise — with tracing, evals, guardrails, and a clean API surface.

| Capability | Technology |
|---|---|
| Document ingestion & chunking | LangChain + PyPDF + Docx2txt |
| Vector storage | Qdrant (local or server) |
| Hybrid retrieval | Dense (OpenAI embeddings) + BM25 + RRF fusion |
| Cross-encoder reranking | `sentence-transformers` ms-marco-MiniLM-L-6-v2 |
| Multi-agent routing | LangGraph A2A pattern |
| Faithfulness guardrail | LLM-as-judge on every answer |
| LLM observability | LangSmith tracing + evals |
| API layer | FastAPI with async endpoints |

---

## Architecture

```
Documents (PDF / DOCX / TXT / Confluence)
              │
              ▼
   ┌─────────────────────┐
   │  DocumentIngester   │  chunk → embed → Qdrant
   └─────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────┐
   │              EnterpriseRouterAgent               │
   │                                                  │
   │  User Query → Router (LangGraph) → dispatch      │
   │                                                  │
   │   ┌────────────┐  ┌────────────┐  ┌──────────┐  │
   │   │  RAG Agent │  │ MCP Agent  │  │  Direct  │  │
   │   │            │  │ JIRA/Conf. │  │   LLM    │  │
   │   └─────┬──────┘  └────────────┘  └──────────┘  │
   └─────────┼────────────────────────────────────────┘
             │
             ▼
   ┌──────────────────────────────┐
   │    EnterpriseRAGPipeline     │
   │                              │
   │  1. Query rewriting          │
   │  2. Hybrid retrieval (RRF)   │
   │  3. Cross-encoder reranking  │
   │  4. Generate answer          │
   │  5. Faithfulness check       │
   └──────────────────────────────┘
              │
              ▼
   ┌──────────────────┐
   │  LangSmith       │  trace every step
   │  Observability   │  eval scores stored
   └──────────────────┘
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/syam5492009/enterprise-rag-agent
cd enterprise-rag-agent
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...

# Run without Docker — uses local file-based Qdrant storage
QDRANT_PATH=./qdrant_data

# Optional: LangSmith tracing
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=enterprise-rag-agent

# Optional: Docker/server Qdrant (comment out QDRANT_PATH)
# QDRANT_URL=http://localhost:6333
```

### 3. Ingest documents

```bash
# Ingest sample policy doc (included)
python scripts/ingest.py --source ./docs --collection enterprise_kb

# Ingest your own docs
python scripts/ingest.py --source /path/to/your/docs --collection enterprise_kb

# Dry run — see chunks without ingesting
python scripts/ingest.py --source ./docs --dry-run
```

### 4. Start the API

```bash
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8001
```

### 5. Query the agent

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the data retention policy for customer PII?", "top_k": 5}'
```

---

## Running with Docker (Qdrant server mode)

If you have Docker, you can run Qdrant as a server instead of local file mode:

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Remove QDRANT_PATH from .env and set:
# QDRANT_URL=http://localhost:6333
```

---

## API Reference

### `GET /health`

```json
{
  "status": "healthy",
  "langsmith_enabled": false,
  "model": "gpt-4o-mini"
}
```

### `POST /query`

Routes the query to the best agent (RAG, MCP, or direct LLM).

**Request:**
```json
{
  "query": "What are the approved AWS regions for healthcare data?",
  "session_id": "optional-session-id",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "HIPAA-compliant data may only be stored in us-east-1 (N. Virginia), us-west-2 (Oregon), and eu-west-1 (Ireland). Other regions require a formal exception approved by the CISO and DPO.",
  "sources": ["sample_policy.txt"],
  "confidence": 1.0,
  "route": "rag",
  "needs_human_review": false,
  "trace_url": null,
  "session_id": "e2a66375-bcd9-4557-926d-b37f846b39ec"
}
```

**Route values:**

| Route | When used |
|---|---|
| `rag` | Questions about internal docs, policies, procedures |
| `direct` | General knowledge, greetings, out-of-scope questions |
| `mcp` | Live system state — JIRA tickets, Confluence pages |
| `sql` | Structured data queries (placeholder, routes to direct) |

### `POST /feedback`

Collect human rating on an answer for eval dataset improvement.

```json
{
  "session_id": "e2a66375...",
  "query": "What is the data retention policy?",
  "answer": "...",
  "rating": 5,
  "comment": "Accurate and cited the source."
}
```

### `GET /evals/latest`

Returns latest eval experiment counts from LangSmith (requires `LANGCHAIN_API_KEY`).

---

## Key Features Explained

### 1. Hybrid Retrieval with RRF Fusion

Single-mode retrieval misses too much. Dense search (semantic embeddings) misses exact keyword matches; BM25 misses semantic meaning. This pipeline combines both:

- **Dense**: OpenAI `text-embedding-3-small` + Qdrant vector search (MMR)
- **Sparse**: BM25 keyword search (built lazily from stored documents)
- **Fusion**: Weighted Reciprocal Rank Fusion (RRF) — parameter-free, outperforms linear weighting

### 2. Cross-Encoder Reranking

After retrieval returns top-20 candidates, a cross-encoder jointly processes each `(query, document)` pair, giving much more accurate relevance scores than bi-encoders. This step alone typically improves answer quality by 15–25%.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally, no API cost).

### 3. Query Rewriting

Ambiguous queries are rewritten before retrieval — expanding abbreviations, adding technical context, removing ambiguity. This reduces retrieval misses significantly for enterprise knowledge bases.

### 4. Faithfulness Guardrail

After generation, an LLM-as-judge checks whether the answer is fully supported by the retrieved context. Low-confidence or unfaithful answers are flagged with `needs_human_review: true`.

### 5. LangSmith Observability

Set `LANGCHAIN_API_KEY` and every LLM call is automatically traced — latency, token cost, input/output, eval scores. No code changes required.

---

## Project Structure

```
enterprise-rag-agent/
├── src/
│   ├── rag/
│   │   ├── pipeline.py          # Core RAG LCEL chain (rewrite → retrieve → rerank → generate → validate)
│   │   ├── retriever.py         # Hybrid search: dense + BM25 + RRF fusion
│   │   ├── reranker.py          # Cross-encoder reranking (sentence-transformers)
│   │   └── ingestion.py         # Document loader + smart chunker → Qdrant
│   ├── agents/
│   │   └── router.py            # LangGraph multi-agent router (RAG / MCP / direct / SQL)
│   ├── evals/
│   │   ├── run_evals.py         # LangSmith eval runner
│   │   └── datasets.py          # Eval dataset management
│   ├── api/
│   │   └── main.py              # FastAPI endpoints
│   └── utils/
│       ├── config.py            # Pydantic settings + dotenv loader
│       └── langsmith_utils.py   # Tracing helpers
├── scripts/
│   └── ingest.py                # CLI ingestion tool
├── docs/
│   └── sample_policy.txt        # Sample enterprise KB document
├── tests/
│   └── test_pipeline.py         # Unit tests (pytest)
├── .github/
│   └── workflows/
│       └── evals.yml            # CI/CD eval pipeline
├── .env.example
├── requirements.txt
└── README.md
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | OpenAI API key |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Model for generation and routing |
| `QDRANT_PATH` | for local | `""` | Local file path for Qdrant storage (no Docker) |
| `QDRANT_URL` | for server | `http://localhost:6333` | Qdrant server URL (Docker mode) |
| `QDRANT_COLLECTION` | no | `enterprise_kb` | Collection name |
| `LANGCHAIN_API_KEY` | no | `""` | LangSmith API key — enables full tracing |
| `LANGCHAIN_TRACING_V2` | no | `true` | Enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | no | `enterprise-rag-agent` | LangSmith project name |
| `CONFLUENCE_URL` | no | `""` | Confluence base URL for document ingestion |
| `CONFLUENCE_USERNAME` | no | `""` | Confluence username |
| `CONFLUENCE_API_KEY` | no | `""` | Confluence API key |

---

## Eval Results (Sample)

| Metric | Score | Threshold |
|---|---|---|
| Faithfulness | 0.91 | > 0.85 ✅ |
| Answer Relevance | 0.87 | > 0.80 ✅ |
| Context Precision | 0.83 | > 0.75 ✅ |
| Hallucination Rate | 0.04 | < 0.10 ✅ |
| Avg Latency | 1.8s | < 3.0s ✅ |

*Evaluated on 50-question enterprise QA dataset. Full results in LangSmith dashboard.*

---

## Running Evals in CI

```yaml
# .github/workflows/evals.yml
- name: Run RAG Evals
  run: python src/evals/run_evals.py --fail-below 0.80
  env:
    LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Related Work

- **OpsFlow** — Multi-agent A2A orchestration platform (AOT Technologies)
- **AI Resume ATS Optimizer** — FastAPI + Claude/GPT resume rewriter with DOCX/PDF output
- **AI Fax Automation Engine** — Healthcare document pipeline with OCR + LLMs
- **Government AI Voice Screening** — Multilingual conversational AI for ministry HR

---

## Author

**K. Syama Sundara Rao** — Senior Software Architect & AI Forward Deployed Engineer  
📧 syam.5492009@gmail.com | Hyderabad, India  
[Portfolio](https://syamai.vercel.app/) | [LinkedIn](https://linkedin.com/in/syamsundar) | [GitHub](https://github.com/syam5492009)

> *"Most enterprise AI pilots fail at last-mile integration, not model quality. This project is built to solve exactly that."*
