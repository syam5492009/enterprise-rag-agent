"""
tests/test_pipeline.py
-----------------------
Unit tests for the RAG pipeline components.
Run with: pytest tests/

These tests use mocks so they run without Qdrant or OpenAI credentials.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.reranker import CrossEncoderReranker
from src.rag.pipeline import RAGResponse, format_docs, extract_sources


# ── Reranker tests ────────────────────────────────────────────────────────────

class TestCrossEncoderReranker:
    def test_rerank_empty_docs_returns_empty(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        result = reranker.rerank("test query", [], top_k=5)
        assert result == []

    def test_rerank_no_model_returns_top_k(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = None
        docs = [Document(page_content=f"doc {i}") for i in range(10)]
        result = reranker.rerank("query", docs, top_k=3)
        assert len(result) == 3
        assert result == docs[:3]

    def test_rerank_with_model_sorts_by_score(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        mock_model = MagicMock()
        # Lower score for first doc, higher for second
        mock_model.predict.return_value = [0.1, 0.9]
        reranker._model = mock_model

        docs = [
            Document(page_content="irrelevant text"),
            Document(page_content="highly relevant text"),
        ]
        result = reranker.rerank("query", docs, top_k=2)

        assert result[0].page_content == "highly relevant text"
        assert result[1].page_content == "irrelevant text"
        assert "rerank_score" in result[0].metadata
        assert result[0].metadata["rerank_position"] == 1

    def test_rerank_adds_position_metadata(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8, 0.6, 0.4]
        reranker._model = mock_model

        docs = [Document(page_content=f"doc {i}") for i in range(3)]
        result = reranker.rerank("query", docs, top_k=3)

        for i, doc in enumerate(result):
            assert doc.metadata["rerank_position"] == i + 1


# ── Pipeline helper tests ─────────────────────────────────────────────────────

class TestPipelineHelpers:
    def test_format_docs_includes_source(self):
        docs = [
            Document(page_content="hello world", metadata={"source": "policy.pdf"}),
        ]
        formatted = format_docs(docs)
        assert "policy.pdf" in formatted
        assert "hello world" in formatted

    def test_format_docs_unknown_source_fallback(self):
        docs = [Document(page_content="no source here")]
        formatted = format_docs(docs)
        assert "Unknown" in formatted

    def test_format_docs_separates_multiple(self):
        docs = [
            Document(page_content="doc one", metadata={"source": "a.pdf"}),
            Document(page_content="doc two", metadata={"source": "b.pdf"}),
        ]
        formatted = format_docs(docs)
        assert "---" in formatted
        assert "doc one" in formatted
        assert "doc two" in formatted

    def test_extract_sources_deduplicates(self):
        docs = [
            Document(page_content="a", metadata={"source": "file.pdf"}),
            Document(page_content="b", metadata={"source": "file.pdf"}),
            Document(page_content="c", metadata={"source": "other.pdf"}),
        ]
        sources = extract_sources(docs)
        assert sorted(sources) == ["file.pdf", "other.pdf"]

    def test_extract_sources_empty(self):
        assert extract_sources([]) == []


# ── RAGResponse schema tests ──────────────────────────────────────────────────

class TestRAGResponse:
    def test_valid_response(self):
        resp = RAGResponse(
            answer="The policy is X.",
            sources=["policy.pdf"],
            confidence=0.92,
            needs_human_review=False,
            query_rewritten="What is the data retention policy?",
        )
        assert resp.confidence == 0.92
        assert not resp.needs_human_review

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            RAGResponse(
                answer="test",
                sources=[],
                confidence=1.5,  # out of range
                needs_human_review=False,
                query_rewritten="test",
            )

    def test_confidence_lower_bound(self):
        with pytest.raises(Exception):
            RAGResponse(
                answer="test",
                sources=[],
                confidence=-0.1,  # out of range
                needs_human_review=False,
                query_rewritten="test",
            )


# ── API smoke test ────────────────────────────────────────────────────────────

def test_health_endpoint():
    """Health endpoint should return without connecting to Qdrant."""
    from fastapi.testclient import TestClient

    # EnterpriseRouterAgent is imported lazily inside the lifespan function,
    # so patch it at the source module, not on src.api.main.
    with patch("src.agents.router.EnterpriseRouterAgent") as mock_agent_cls:
        mock_agent_cls.return_value = MagicMock()
        from src.api.main import app
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "model" in data
