"""
src/rag/retriever.py
---------------------
Hybrid retriever combining dense (semantic) + sparse (BM25) search.
This is what separates production RAG from demo RAG — single-mode retrieval
misses too much. Hybrid search is now the enterprise standard.

Results from both retrievers are merged with Reciprocal Rank Fusion (RRF),
which is a parameter-free fusion method that outperforms linear weighting.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from typing import List
import logging

from src.utils.config import settings

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    results_lists: list[list[Document]],
    weights: list[float],
    k: int = 60,
) -> list[Document]:
    """
    Merge ranked lists using weighted Reciprocal Rank Fusion.

    RRF score for a doc = sum over lists of: weight / (rank + k)
    k=60 is the standard constant that prevents very early ranks from
    dominating. Final list is sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    doc_store: dict[str, Document] = {}

    for results, weight in zip(results_lists, weights):
        for rank, doc in enumerate(results):
            key = doc.page_content[:200]  # use content prefix as dedup key
            scores[key] = scores.get(key, 0.0) + weight / (rank + k)
            doc_store[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_store[key] for key in sorted_keys]


class HybridRetriever:
    """
    Hybrid retriever combining:
    - Dense: OpenAI embeddings + Qdrant vector search (MMR)
    - Sparse: BM25 keyword search
    - Fusion: Weighted Reciprocal Rank Fusion

    Why hybrid? Semantic search misses exact keyword matches (product names,
    IDs, technical terms). BM25 misses semantic meaning. Together they cover
    both failure modes — critical for enterprise knowledge bases.
    """

    def __init__(
        self,
        collection_name: str = "enterprise_kb",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self.collection_name = collection_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.qdrant_client = (
            QdrantClient(path=settings.QDRANT_PATH)
            if settings.QDRANT_PATH
            else QdrantClient(url=settings.QDRANT_URL)
        )

        self._vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
        # BM25 retriever is built lazily from loaded documents
        self._bm25_retriever: BM25Retriever | None = None

    def _build_bm25(self, top_k: int) -> BM25Retriever | None:
        """Build BM25 index from Qdrant documents on first use."""
        if self._bm25_retriever is not None:
            self._bm25_retriever.k = top_k
            return self._bm25_retriever

        logger.info("Building BM25 index from vector store...")
        all_points = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
        )[0]

        docs = [
            Document(
                page_content=p.payload.get("page_content", ""),
                metadata=p.payload.get("metadata", {})
            )
            for p in all_points
            if p.payload.get("page_content")
        ]

        if not docs:
            logger.warning("No documents found for BM25 index")
            return None

        self._bm25_retriever = BM25Retriever.from_documents(docs)
        self._bm25_retriever.k = top_k
        logger.info("BM25 index built with %d documents", len(docs))
        return self._bm25_retriever

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Retrieve documents using hybrid search (RRF fusion).

        Args:
            query: The search query (should be pre-processed/rewritten)
            top_k: Number of documents to return after fusion

        Returns:
            List of ranked Document objects
        """
        try:
            dense_retriever = self._vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k, "fetch_k": top_k * 3}
            )
            dense_docs = dense_retriever.invoke(query)

            bm25 = self._build_bm25(top_k)
            if bm25 is not None:
                sparse_docs = bm25.invoke(query)
                merged = _reciprocal_rank_fusion(
                    [dense_docs, sparse_docs],
                    [self.dense_weight, self.sparse_weight],
                )
                result = merged[:top_k]
                logger.info(
                    "Hybrid retrieval: dense=%d, sparse=%d → merged=%d",
                    len(dense_docs), len(sparse_docs), len(result)
                )
                return result

            # BM25 unavailable (empty collection) — fall back to dense only
            logger.warning("BM25 unavailable, using dense retrieval only")
            return dense_docs[:top_k]

        except Exception as e:
            logger.error("Hybrid retrieval failed, falling back to dense: %s", e)
            return self._vector_store.similarity_search(query, k=top_k)

    def similarity_search_with_score(self, query: str, k: int = 5):
        """Direct similarity search with scores, useful for eval debugging."""
        return self._vector_store.similarity_search_with_relevance_scores(query, k=k)
