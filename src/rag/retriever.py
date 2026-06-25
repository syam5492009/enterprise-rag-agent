"""
src/rag/retriever.py
---------------------
Hybrid retriever combining dense (semantic) + sparse (BM25) search.
This is what separates production RAG from demo RAG — single-mode retrieval
misses too much. Hybrid search is now the enterprise standard.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from typing import List
import logging

from src.utils.config import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining:
    - Dense: OpenAI embeddings + Qdrant vector search
    - Sparse: BM25 keyword search
    - Ensemble: weighted combination of both
    
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
        self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)

        self._vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
        # BM25 retriever is built lazily from loaded documents
        self._bm25_retriever: BM25Retriever | None = None
        self._ensemble_retriever: EnsembleRetriever | None = None

    def _build_ensemble(self, top_k: int) -> EnsembleRetriever:
        """Build the ensemble retriever. Called on first use."""
        dense_retriever = self._vector_store.as_retriever(
            search_type="mmr",  # Max Marginal Relevance for diversity
            search_kwargs={"k": top_k, "fetch_k": top_k * 3}
        )

        if self._bm25_retriever is None:
            # Load all docs from Qdrant to build BM25 index
            # In production, cache this and rebuild on ingestion
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

            if docs:
                self._bm25_retriever = BM25Retriever.from_documents(docs)
                self._bm25_retriever.k = top_k
                logger.info("BM25 index built with %d documents", len(docs))
            else:
                logger.warning("No documents found for BM25 index")
                return dense_retriever  # Fallback to dense only

        self._bm25_retriever.k = top_k

        return EnsembleRetriever(
            retrievers=[dense_retriever, self._bm25_retriever],
            weights=[self.dense_weight, self.sparse_weight]
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: The search query (should be pre-processed/rewritten)
            top_k: Number of documents to return
            
        Returns:
            List of ranked Document objects
        """
        try:
            ensemble = self._build_ensemble(top_k)
            docs = ensemble.invoke(query)
            logger.info("Hybrid retrieval returned %d documents", len(docs))
            return docs

        except Exception as e:
            logger.error("Hybrid retrieval failed, falling back to dense: %s", e)
            # Graceful degradation — always return something
            return self._vector_store.similarity_search(query, k=top_k)

    def similarity_search_with_score(self, query: str, k: int = 5):
        """Direct similarity search with scores, useful for eval debugging."""
        return self._vector_store.similarity_search_with_relevance_scores(query, k=k)
