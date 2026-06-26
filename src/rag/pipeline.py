"""
src/rag/pipeline.py
-------------------
Production-grade RAG pipeline using LangChain LCEL.
Includes query rewriting, hybrid retrieval, reranking, and self-validation.
Every step is traced via LangSmith automatically.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from src.rag.retriever import HybridRetriever
from src.rag.reranker import CrossEncoderReranker
from src.utils.config import settings

logger = logging.getLogger(__name__)


# ── Output schema ────────────────────────────────────────────────────────────

class RAGResponse(BaseModel):
    """Structured response from the RAG pipeline."""
    answer: str = Field(description="The answer to the user's question")
    sources: List[str] = Field(description="Source document names used")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)
    needs_human_review: bool = Field(description="Flag for low-confidence answers")
    query_rewritten: str = Field(description="The rewritten query used for retrieval")


# ── Prompts ───────────────────────────────────────────────────────────────────

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at reformulating user queries for better document retrieval.
    
Rewrite the query to be more specific and retrieval-friendly.
- Expand abbreviations
- Add relevant technical context  
- Remove ambiguity
- Keep it concise (1-2 sentences max)

Return ONLY the rewritten query, nothing else."""),
    ("human", "Original query: {query}")
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise enterprise knowledge assistant. 
    
Answer questions ONLY based on the provided context. 

Rules:
1. If the context doesn't contain enough information, say "I don't have sufficient information in the knowledge base to answer this."
2. Never hallucinate or make up facts
3. Cite which document(s) your answer comes from
4. Be concise and professional

Context:
{context}

Sources available: {sources}"""),
    ("human", "{question}")
])

FAITHFULNESS_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a faithfulness evaluator. Check if the answer is fully supported by the context.

Answer with a JSON object:
{{"is_faithful": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}

Only respond with the JSON, nothing else."""),
    ("human", """Context: {context}

Answer to evaluate: {answer}

Is this answer fully supported by the context?""")
])


# ── Helper functions ──────────────────────────────────────────────────────────

def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for the prompt."""
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def extract_sources(docs: List[Document]) -> List[str]:
    """Extract unique source names from retrieved documents."""
    return list({doc.metadata.get("source", "Unknown") for doc in docs})


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class EnterpriseRAGPipeline:
    """
    Production RAG pipeline with:
    - Query rewriting for better retrieval
    - Hybrid search (semantic + BM25)
    - Cross-encoder reranking
    - Self-RAG faithfulness validation
    - LangSmith tracing (automatic via env vars)
    """

    def __init__(self, collection_name: str = "enterprise_kb"):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
        )
        self.retriever = HybridRetriever(collection_name=collection_name)
        self.reranker = CrossEncoderReranker()

        # Build chains
        self._query_rewriter = self._build_query_rewriter()
        self._rag_chain = self._build_rag_chain()

    def _build_query_rewriter(self):
        return (
            QUERY_REWRITE_PROMPT
            | self.llm.with_config({"run_name": "query_rewriter"})
            | StrOutputParser()
        )

    def _build_rag_chain(self):
        return (
            RAG_PROMPT
            | self.llm.with_config({"run_name": "rag_generator"})
            | StrOutputParser()
        )

    def _check_faithfulness(self, answer: str, context: str) -> dict:
        """LLM-as-judge faithfulness check — core guardrail."""
        import json
        chain = (
            FAITHFULNESS_CHECK_PROMPT
            | self.llm.with_config({"run_name": "faithfulness_judge"})
            | StrOutputParser()
        )
        result = chain.invoke({"context": context, "answer": answer})
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.warning("Faithfulness check returned non-JSON: %s", result)
            return {"is_faithful": True, "confidence": 0.5, "reason": "parse error"}

    def invoke(
        self,
        query: str,
        top_k: int = 5,
        rewrite_query: bool = True,
        check_faithfulness: bool = True,
    ) -> RAGResponse:
        """
        Full RAG pipeline invocation.

        Steps:
        1. (Optional) Rewrite query for better retrieval
        2. Hybrid retrieval (semantic + BM25)
        3. Cross-encoder reranking
        4. Generate answer
        5. (Optional) Faithfulness validation
        """
        logger.info("RAG pipeline invoked: %s", query[:80])

        # Step 1: Query rewriting (optional — skip to save ~1-2s)
        if rewrite_query:
            rewritten_query = self._query_rewriter.invoke({"query": query})
            logger.info("Rewritten query: %s", rewritten_query)
        else:
            rewritten_query = query
            logger.info("Query rewriting skipped")

        # Step 2: Hybrid retrieval
        docs = self.retriever.retrieve(rewritten_query, top_k=top_k * 2)

        # Step 3: Reranking
        reranked_docs = self.reranker.rerank(rewritten_query, docs, top_k=top_k)

        # Step 4: Format context and generate
        context = format_docs(reranked_docs)
        sources = extract_sources(reranked_docs)

        answer = self._rag_chain.invoke({
            "context": context,
            "question": query,
            "sources": ", ".join(sources)
        })

        # Step 5: Faithfulness check (optional — skip to save ~1-2s)
        if check_faithfulness:
            faithfulness = self._check_faithfulness(answer, context)
            confidence = faithfulness.get("confidence", 0.8)
            needs_review = not faithfulness.get("is_faithful", True) or confidence < 0.6
            if needs_review:
                logger.warning(
                    "Answer flagged for review. Faithful: %s, Confidence: %.2f",
                    faithfulness.get("is_faithful"),
                    confidence,
                )
        else:
            confidence = 0.8
            needs_review = False
            logger.info("Faithfulness check skipped")

        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            needs_human_review=needs_review,
            query_rewritten=rewritten_query,
        )

    async def ainvoke(self, query: str, top_k: int = 5) -> RAGResponse:
        """Async version for FastAPI endpoints."""
        import asyncio
        return await asyncio.to_thread(self.invoke, query, top_k)

    async def astream(
        self,
        query: str,
        top_k: int = 5,
        rewrite_query: bool = False,
        check_faithfulness: bool = False,
    ):
        """
        Async generator that streams LLM tokens then emits a final metadata event.
        Yields dicts: {"type": "token", "content": "..."} then {"type": "done", ...}
        """
        import asyncio

        # Retrieval runs in a thread (blocking I/O)
        if rewrite_query:
            rewritten_query = await asyncio.to_thread(
                self._query_rewriter.invoke, {"query": query}
            )
            logger.info("Rewritten query: %s", rewritten_query)
        else:
            rewritten_query = query

        docs = await asyncio.to_thread(self.retriever.retrieve, rewritten_query, top_k * 2)
        reranked_docs = await asyncio.to_thread(self.reranker.rerank, rewritten_query, docs, top_k)

        context = format_docs(reranked_docs)
        sources = extract_sources(reranked_docs)

        # Stream generation tokens
        chain = RAG_PROMPT | self.llm.with_config({"run_name": "rag_generator"})
        full_answer = []
        async for chunk in chain.astream({
            "context": context,
            "question": query,
            "sources": ", ".join(sources),
        }):
            token = chunk.content
            if token:
                full_answer.append(token)
                yield {"type": "token", "content": token}

        # Faithfulness check after full answer is assembled (optional)
        if check_faithfulness:
            faithfulness = await asyncio.to_thread(
                self._check_faithfulness, "".join(full_answer), context
            )
            confidence = faithfulness.get("confidence", 0.8)
            needs_review = not faithfulness.get("is_faithful", True) or confidence < 0.6
        else:
            confidence = 0.8
            needs_review = False

        yield {
            "type": "done",
            "sources": sources,
            "confidence": confidence,
            "needs_human_review": needs_review,
            "route": "rag",
        }
