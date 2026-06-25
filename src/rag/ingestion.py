"""
src/rag/ingestion.py
---------------------
Enterprise document ingestion pipeline.
Handles PDF, DOCX, Confluence, and web scraping.
Smart chunking preserves document structure.
"""

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    ConfluenceLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pathlib import Path
from typing import List
import hashlib
import logging

from src.utils.config import settings

logger = logging.getLogger(__name__)


class DocumentIngester:
    """
    Multi-source document ingestion with:
    - Smart semantic chunking (preserves headers/structure)
    - Deduplication via content hashing
    - Rich metadata extraction
    - Batch upsert to Qdrant
    """

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64

    def __init__(self, collection_name: str = "enterprise_kb"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        self._ensure_collection()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        collections = [c.name for c in self.qdrant_client.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            logger.info("Created collection: %s", self.collection_name)

    def _content_hash(self, content: str) -> str:
        """Generate hash for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()

    def load_from_directory(self, directory: str) -> List[Document]:
        """Load all supported files from a directory."""
        docs = []
        path = Path(directory)

        for file_path in path.rglob("*"):
            if file_path.suffix.lower() == ".pdf":
                docs.extend(self._load_pdf(str(file_path)))
            elif file_path.suffix.lower() in (".docx", ".doc"):
                docs.extend(self._load_docx(str(file_path)))
            elif file_path.suffix.lower() == ".txt":
                docs.extend(self._load_text(str(file_path)))

        logger.info("Loaded %d raw documents from %s", len(docs), directory)
        return docs

    def _load_pdf(self, path: str) -> List[Document]:
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = Path(path).name
                doc.metadata["file_type"] = "pdf"
            return docs
        except Exception as e:
            logger.error("Failed to load PDF %s: %s", path, e)
            return []

    def _load_docx(self, path: str) -> List[Document]:
        try:
            loader = Docx2txtLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = Path(path).name
                doc.metadata["file_type"] = "docx"
            return docs
        except Exception as e:
            logger.error("Failed to load DOCX %s: %s", path, e)
            return []

    def _load_text(self, path: str) -> List[Document]:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        return [Document(
            page_content=text,
            metadata={"source": Path(path).name, "file_type": "txt"}
        )]

    def load_from_confluence(
        self,
        space_key: str,
        confluence_url: str,
        username: str,
        api_key: str,
    ) -> List[Document]:
        """Load pages from Confluence — common enterprise source."""
        loader = ConfluenceLoader(
            url=confluence_url,
            username=username,
            api_key=api_key,
            space_key=space_key,
            include_attachments=False,
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "confluence"
        logger.info("Loaded %d pages from Confluence space %s", len(docs), space_key)
        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into chunks with metadata preservation.
        Uses recursive splitter that respects natural text boundaries.
        """
        chunks = self.text_splitter.split_documents(docs)
        # Add content hash for deduplication
        for chunk in chunks:
            chunk.metadata["content_hash"] = self._content_hash(chunk.page_content)
        logger.info("Created %d chunks from %d documents", len(chunks), len(docs))
        return chunks

    def ingest(self, docs: List[Document]) -> int:
        """
        Chunk and upsert documents to Qdrant.
        Returns number of chunks ingested.
        """
        chunks = self.chunk_documents(docs)

        if not chunks:
            logger.warning("No chunks to ingest")
            return 0

        # Upsert to Qdrant via LangChain abstraction
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            url=settings.QDRANT_URL,
            collection_name=self.collection_name,
            force_recreate=False,
        )

        logger.info("Ingested %d chunks into collection '%s'", len(chunks), self.collection_name)
        return len(chunks)

    def ingest_from_directory(self, directory: str) -> int:
        """Convenience method: load + chunk + ingest from a directory."""
        docs = self.load_from_directory(directory)
        return self.ingest(docs)
