"""
scripts/ingest.py
------------------
CLI tool for ingesting documents into the knowledge base.
Run: python scripts/ingest.py --source ./docs --collection enterprise_kb
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Enterprise RAG knowledge base"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to directory containing documents (PDF, DOCX, TXT)"
    )
    parser.add_argument(
        "--collection",
        default="enterprise_kb",
        help="Qdrant collection name (default: enterprise_kb)"
    )
    parser.add_argument(
        "--confluence-space",
        help="Confluence space key to ingest (optional)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk documents without ingesting to Qdrant"
    )
    args = parser.parse_args()

    # Validate source
    source_path = Path(args.source)
    if not source_path.exists():
        logger.error("Source path does not exist: %s", args.source)
        sys.exit(1)

    from src.rag.ingestion import DocumentIngester

    ingester = DocumentIngester(collection_name=args.collection)

    # Load from directory
    logger.info("Loading documents from: %s", args.source)
    docs = ingester.load_from_directory(args.source)

    if not docs:
        logger.warning("No documents found in %s", args.source)
        sys.exit(0)

    if args.dry_run:
        chunks = ingester.chunk_documents(docs)
        logger.info(
            "DRY RUN: Would ingest %d chunks from %d documents",
            len(chunks), len(docs)
        )
        for i, chunk in enumerate(chunks[:3]):
            logger.info("Sample chunk %d: %s...", i + 1, chunk.page_content[:100])
        return

    # Ingest
    count = ingester.ingest(docs)
    logger.info(
        "✅ Ingestion complete: %d chunks → collection '%s'",
        count, args.collection
    )


if __name__ == "__main__":
    main()
