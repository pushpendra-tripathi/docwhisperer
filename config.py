"""Configuration module for the RAG application.

This module provides configuration settings for the RAG application,
with support for environment variables and type-safe configuration.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the RAG application."""

    # Storage paths
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "chroma")
    DATA_PATH: str = os.getenv("DATA_PATH", "data")

    # Model configurations
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "deepseek-r1:1.5b")
    # DEFAULT_MODEL: str = "llama3.2"
    # EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "snowflake-arctic-embed2")

    # Chunking configurations
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "10"))

    # LLM configurations
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

    def __post_init__(self) -> None:
        """Validate configuration settings after initialization."""
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if self.TOP_K_RESULTS <= 0:
            raise ValueError("TOP_K_RESULTS must be positive")
        if not 0 <= self.TEMPERATURE <= 1:
            raise ValueError("TEMPERATURE must be between 0 and 1")


# Create a global config instance
config = Config()
