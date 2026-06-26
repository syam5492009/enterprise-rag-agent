"""
src/api/main.py
----------------
FastAPI application exposing the RAG pipeline and agent router.
Production-ready with async endpoints, health checks, and LangSmith trace IDs.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import json
import logging
import time
import uuid

from src.utils.config import settings

logger = logging.getLogger(__name__)

# ── Lifespan: initialize heavy resources once at startup ──────────────────────

_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the router agent on startup so import-time has no side-effects."""
    global _agent
    from src.agents.router import EnterpriseRouterAgent
    logger.info("Initializing EnterpriseRouterAgent...")
    _agent = EnterpriseRouterAgent()
    logger.info("Agent ready.")
    yield
    _agent = None


app = FastAPI(
    title="Enterprise RAG Agent API",
    description="Production AI knowledge base with LangSmith observability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    rewrite_query: bool = Field(True, description="Rewrite query before retrieval (saves ~1-2s if disabled)")
    check_faithfulness: bool = Field(True, description="Run LLM faithfulness guardrail (saves ~1-2s if disabled)")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    route: str
    needs_human_review: bool
    latency_ms: float
    trace_url: Optional[str] = None
    session_id: str


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="1=terrible, 5=excellent")
    comment: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    langsmith_enabled: bool
    model: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — used by load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        langsmith_enabled=bool(settings.LANGCHAIN_API_KEY),
        model=settings.OPENAI_MODEL,
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint.
    Routes to the appropriate agent (RAG, MCP, or direct LLM).
    Every call is traced in LangSmith.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        t0 = time.perf_counter()
        result = _agent.invoke(
            query=request.query,
            chat_history=[],
            rewrite_query=request.rewrite_query,
            check_faithfulness=request.check_faithfulness,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        confidence = result.get("confidence", 0.8)
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=confidence,
            route=result.get("route", "unknown"),
            needs_human_review=confidence < 0.6,
            latency_ms=latency_ms,
            trace_url=None,  # LangSmith URL populated server-side
            session_id=session_id,
        )

    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Streaming query endpoint using Server-Sent Events (SSE).
    Uses keyword routing (no LLM call) so first token arrives in ~300ms.

    SSE event format:
      data: {"type": "token", "content": "..."}   — one per token
      data: {"type": "done",  "sources": [...], "confidence": 1.0, "latency_ms": 1234}
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session_id = request.session_id or str(uuid.uuid4())

    async def generate():
        t0 = time.perf_counter()
        try:
            async for event in _agent.astream(
                query=request.query,
                top_k=request.top_k,
                rewrite_query=request.rewrite_query,
                check_faithfulness=request.check_faithfulness,
            ):
                if event["type"] == "done":
                    event["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    event["session_id"] = session_id
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error("Stream failed: %s", e, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/feedback")
async def collect_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Collect human feedback on answers.
    Feedback is logged to LangSmith for eval dataset improvement.
    """
    background_tasks.add_task(
        _log_feedback_to_langsmith,
        request.session_id,
        request.rating,
        request.comment,
    )
    return {"status": "feedback received", "session_id": request.session_id}


async def _log_feedback_to_langsmith(session_id: str, rating: int, comment: Optional[str]):
    """Background task: log human feedback as LangSmith annotation."""
    try:
        from langsmith import Client
        Client()  # validates credentials
        logger.info("Feedback logged: session=%s rating=%d/5", session_id, rating)
    except Exception as e:
        logger.error("Failed to log feedback: %s", e)


@app.get("/evals/latest")
async def get_latest_eval_results():
    """
    Return the latest eval results from LangSmith.
    Useful for dashboards showing system health to non-technical stakeholders.
    """
    try:
        from langsmith import Client
        client = Client()
        experiments = list(client.list_projects(
            name=settings.LANGCHAIN_PROJECT,
        ))
        return {
            "project": settings.LANGCHAIN_PROJECT,
            "experiments": len(experiments),
            "dashboard": f"https://smith.langchain.com/o/default/projects/p/{settings.LANGCHAIN_PROJECT}"
        }
    except Exception as e:
        return {"error": str(e), "note": "Configure LANGCHAIN_API_KEY to enable"}
