"""Document processing module for RAG system.

This module handles document loading, chunking, and database population
with support for parallel processing and progress tracking.
"""

import argparse
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from config import config
from get_embedding_function import get_embedding_function

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MAX_WORKERS = 4
BATCH_SIZE = 100


@dataclass
class ProcessingStats:
    """Statistics for document processing."""

    total_documents: int = 0
    total_chunks: int = 0
    new_chunks: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message to the stats."""
        self.errors.append(error)

    def __str__(self) -> str:
        """Format statistics as a string."""
        return (
            f"Processing Statistics:\n"
            f"- Total Documents: {self.total_documents}\n"
            f"- Total Chunks: {self.total_chunks}\n"
            f"- New Chunks Added: {self.new_chunks}\n"
            f"- Processing Time: {self.processing_time:.2f}s\n"
            f"- Errors: {len(self.errors)}"
        )


class DocumentProcessor:
    """Handles document processing and database management."""

    def __init__(self):
        """Initialize the document processor."""
        self.embedding_function = get_embedding_function()
        self.db: Optional[Chroma] = None
        self.stats = ProcessingStats()

    def initialize_database(self) -> None:
        """Initialize or connect to the existing database.

        Raises:
            Exception: If database initialization fails
        """
        try:
            # Ensure directory exists with proper permissions
            chroma_dir = Path(config.CHROMA_PATH)
            chroma_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

            # Add retry logic for connection
            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    self.db = Chroma(
                        persist_directory=str(chroma_dir),
                        embedding_function=self.embedding_function,
                    )

                    # Verify connection with timeout
                    if not self.db._client.heartbeat():
                        raise ConnectionError("Chroma heartbeat failed")

                    logger.info("Database initialized successfully")
                    return

                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Connection attempt {retry_count} failed: {str(e)}")
                    time.sleep(0.5 * retry_count)

            raise RuntimeError("Failed to initialize database after multiple attempts")

        except Exception as e:
            self.db = None
            logger.error(f"Database initialization failed: {str(e)}")
            raise

    def load_documents(self) -> List[Document]:
        """Load documents from the data directory.

        Returns:
            List of loaded documents

        Raises:
            Exception: If document loading fails
        """
        try:
            data_path = Path(config.DATA_PATH)
            if not data_path.exists():
                logger.warning(f"Data directory {data_path} does not exist")
                return []

            logger.info(f"Loading documents from {data_path}")
            document_loader = PyPDFDirectoryLoader(str(data_path))
            documents = document_loader.load()

            self.stats.total_documents = len(documents)
            logger.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            error_msg = f"Failed to load documents: {str(e)}"
            logger.error(error_msg)
            self.stats.add_error(error_msg)
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with progress tracking.

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks

        Raises:
            Exception: If document splitting fails
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )

            # Process documents in parallel
            chunks = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for doc in documents:
                    future = executor.submit(text_splitter.split_documents, [doc])
                    futures.append(future)

                # Track progress
                with tqdm(total=len(documents), desc="Splitting documents") as pbar:
                    for future in as_completed(futures):
                        try:
                            doc_chunks = future.result()
                            chunks.extend(doc_chunks)
                            pbar.update(1)
                        except Exception as e:
                            error_msg = f"Error splitting document: {str(e)}"
                            logger.error(error_msg)
                            self.stats.add_error(error_msg)

            self.stats.total_chunks = len(chunks)
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            error_msg = f"Failed to split documents: {str(e)}"
            logger.error(error_msg)
            self.stats.add_error(error_msg)
            raise

    def calculate_chunk_ids(self, chunks: List[Document]) -> List[Document]:
        """Calculate unique IDs for each chunk with metadata enhancement.

        Args:
            chunks: List of document chunks

        Returns:
            List of chunks with IDs and enhanced metadata

        Raises:
            Exception: If ID calculation fails
        """
        try:
            chunk_mapping: Dict[str, int] = {}

            for chunk in chunks:
                # Enhance metadata
                source = chunk.metadata.get("source", "unknown")
                page = chunk.metadata.get("page", 0)
                chunk.metadata.update(
                    {
                        "file_name": Path(source).name,
                        "file_type": Path(source).suffix.lower(),
                        "page_number": page,
                        "chunk_size": len(chunk.page_content),
                    }
                )

                # Calculate chunk ID
                page_id = f"{source}:{page}"
                current_count = chunk_mapping.get(page_id, 0)
                chunk.metadata["id"] = f"{page_id}:{current_count}"
                chunk_mapping[page_id] = current_count + 1

            return chunks
        except Exception as e:
            error_msg = f"Failed to calculate chunk IDs: {str(e)}"
            logger.error(error_msg)
            self.stats.add_error(error_msg)
            raise

    def add_to_chroma(self, chunks: List[Document]) -> None:
        """Add new documents to the database with batching and progress tracking."""
        try:
            if not self.db:
                self.initialize_database()
                if not self.db:
                    raise RuntimeError("Database instance creation failed")

            # Get existing IDs
            existing_items = self.db.get(include=[])
            existing_ids = set(existing_items["ids"])

            # Filter new chunks
            new_chunks = [
                chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids
            ]

            if not new_chunks:
                logger.info("No new documents to add")
                return

            # Process in batches with progress tracking
            self.stats.new_chunks = len(new_chunks)
            logger.info(f"Adding {len(new_chunks)} new chunks")

            for i in tqdm(
                range(0, len(new_chunks), BATCH_SIZE), desc="Adding to database"
            ):
                batch = new_chunks[i : i + BATCH_SIZE]
                batch_ids = [chunk.metadata["id"] for chunk in batch]
                self.db.add_documents(batch, ids=batch_ids)

            logger.info("Successfully added new documents")
        except Exception as e:
            error_msg = f"Failed to add documents to database: {str(e)}"
            logger.error(error_msg)
            self.stats.add_error(error_msg)
            raise

    def process_documents(self, reset: bool = False) -> ProcessingStats:
        """Main processing pipeline with statistics tracking.

        Args:
            reset: Whether to reset the database before processing

        Returns:
            Processing statistics

        Raises:
            Exception: If processing fails
        """
        import time

        start_time = time.time()

        try:
            if reset:
                self.clear_database()

            documents = self.load_documents()
            if documents:
                chunks = self.split_documents(documents)
                chunks = self.calculate_chunk_ids(chunks)
                self.add_to_chroma(chunks)

            self.stats.processing_time = time.time() - start_time
            logger.info(str(self.stats))
            return self.stats

        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
            logger.error(error_msg)
            self.stats.add_error(error_msg)
            raise

    def clear_database(self) -> None:
        """Clear the existing database and reset the database reference.

        Raises:
            Exception: If database clearing fails
        """
        try:
            db_path = Path(config.CHROMA_PATH)
            if db_path.exists():
                shutil.rmtree(db_path)
                logger.info("Database cleared successfully")
            self.db = None  # Reset the database reference
        except Exception as e:
            error_msg = f"Failed to clear database: {str(e)}"
            logger.error(error_msg)
            raise


def main():
    """Main entry point with argument parsing and error handling."""
    try:
        parser = argparse.ArgumentParser(
            description="Process documents and populate the vector database."
        )
        parser.add_argument(
            "--reset", action="store_true", help="Reset the database before processing"
        )
        args = parser.parse_args()

        processor = DocumentProcessor()
        stats = processor.process_documents(reset=args.reset)

        if stats.errors:
            logger.warning("Processing completed with errors:")
            for error in stats.errors:
                logger.warning(f"- {error}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
