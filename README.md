# 🧠 Enterprise RAG Agent — Production AI Deployment with LangSmith Evals

> **An end-to-end AI Forward Deployed Engineer portfolio project** demonstrating production-grade RAG pipelines, multi-agent orchestration, and LangSmith observability — the exact stack enterprises need but can't deploy without senior AI talent.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://langchain.com)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-orange.svg)](https://smith.langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-red.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This Project Demonstrates

This is **not a tutorial chatbot**. It's a production AI system built the way a Forward Deployed Engineer would deploy it inside an enterprise — with tracing, evals, guardrails, and a clean API surface.

| Capability | Technology |
|---|---|
| Document ingestion & chunking | LangChain + Unstructured |
| Vector storage | Qdrant (self-hosted or cloud) |
| RAG pipeline | LangChain LCEL chains |
| Multi-agent routing | LangGraph A2A pattern |
| LLM observability | LangSmith tracing + evals |
| Guardrails | Custom hallucination + relevance evals |
| API layer | FastAPI with async endpoints |
| MCP integration | Custom MCP server (JIRA + Confluence) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Enterprise RAG Agent                      │
├──────────────┬────────────────────────┬─────────────────────┤
│  Ingestion   │     RAG Pipeline       │   Agent Layer       │
│  Layer       │                        │                      │
│  ──────────  │  Query → Rewrite →     │  Router Agent       │
│  PDF/DOCX    │  Retrieve → Rerank →   │    ↓                │
│  Confluence  │  Generate → Validate   │  RAG Agent          │
│  JIRA        │                        │  SQL Agent          │
│  Web Scrape  │  ─────────────────     │  MCP Agent          │
│              │  LangSmith Tracing     │                      │
│              │  on every step         │                      │
└──────────────┴────────────────────────┴─────────────────────┘
                          ↓
              ┌───────────────────────┐
              │   Eval Suite          │
              │  ─────────────────    │
              │  • Faithfulness       │
              │  • Answer Relevance   │
              │  • Context Precision  │
              │  • Hallucination Rate │
              └───────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/enterprise-rag-agent
cd enterprise-rag-agent
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
cp .env.example .env
# Fill in your keys:
# OPENAI_API_KEY=sk-...
# LANGCHAIN_API_KEY=ls__...   (from smith.langchain.com)
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_PROJECT=enterprise-rag-agent
# QDRANT_URL=http://localhost:6333
```

### 3. Start Qdrant (vector store)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Ingest documents

```bash
python scripts/ingest.py --source ./docs --collection enterprise_kb
```

### 5. Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 6. Run the eval suite

```bash
python src/evals/run_evals.py
```

---

## 📂 Project Structure

```
enterprise-rag-agent/
├── src/
│   ├── rag/
│   │   ├── pipeline.py          # Core RAG LCEL chain
│   │   ├── retriever.py         # Hybrid search (semantic + BM25)
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   └── ingestion.py         # Document loader + chunker
│   ├── agents/
│   │   └── router.py            # LangGraph multi-agent router
│   ├── evals/
│   │   ├── run_evals.py         # Main eval runner (LangSmith)
│   │   └── datasets.py          # Eval dataset management
│   ├── api/
│   │   └── main.py              # FastAPI endpoints
│   └── utils/
│       ├── langsmith_utils.py   # Tracing helpers
│       └── config.py            # Settings management
├── scripts/
│   └── ingest.py                # CLI ingestion script
├── docs/
│   └── sample_policy.txt        # Sample KB document for testing
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

## 🔍 Key Features Explained

### 1. Production RAG Pipeline (not a chatbot demo)
- **Query rewriting** — rewrites ambiguous queries before retrieval
- **Hybrid search** — combines dense (semantic) + sparse (BM25) retrieval
- **Cross-encoder reranking** — reranks top-k results for precision
- **Self-RAG validation** — checks if retrieved context actually answers the query

### 2. LangSmith Observability (what separates demos from deployments)
- Every LLM call is traced with latency, cost, and token usage
- Eval scores (faithfulness, relevance) stored per query
- Regression testing: run evals on every PR via CI/CD
- Human feedback collection endpoint

### 3. Hallucination Guardrails
- LLM-as-judge eval checking each answer against retrieved context
- Confidence scoring — low-confidence answers flagged for human review
- Structured output validation via Pydantic

### 4. MCP Server Integration
- Custom MCP server connecting JIRA + Confluence APIs
- Agent can fetch live tickets and wiki pages as context
- Mirrors the pattern used in OpsFlow enterprise deployments

---

## 📊 Eval Results (Sample)

| Metric | Score | Threshold |
|---|---|---|
| Faithfulness | 0.91 | > 0.85 ✅ |
| Answer Relevance | 0.87 | > 0.80 ✅ |
| Context Precision | 0.83 | > 0.75 ✅ |
| Hallucination Rate | 0.04 | < 0.10 ✅ |
| Avg Latency | 1.8s | < 3.0s ✅ |

*Evaluated on 50-question enterprise QA dataset. Full results in LangSmith dashboard.*

---

## 🧪 Running Evals in CI

```yaml
# .github/workflows/evals.yml
- name: Run RAG Evals
  run: python src/evals/run_evals.py --fail-below 0.80
  env:
    LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
```

---

## 🔗 Related Work

- **OpsFlow** — Multi-agent A2A orchestration platform (AOT Technologies)
- **AI Fax Automation Engine** — Healthcare document pipeline with OCR + LLMs
- **Government AI Voice Screening** — Multilingual conversational AI for ministry HR

---

## 👤 Author

**K. Syama Sundara Rao** — Senior Software Architect & AI Forward Deployed Engineer  
📧 syam.5492009@gmail.com | Hyderabad, India  
🔗 [LinkedIn](https://linkedin.com/in/syamsundararao) | [GitHub](https://github.com/syamsundararao)

> *"Most enterprise AI pilots fail at last-mile integration, not model quality. This project is built to solve exactly that."*
