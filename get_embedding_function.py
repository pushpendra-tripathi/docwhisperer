"""Embedding function provider for document processing.

This module provides a factory for creating and managing embedding functions,
supporting multiple embedding providers with automatic fallback and caching.
"""

import logging
import os
from enum import Enum
from functools import lru_cache
from typing import Callable, Dict, Optional, Type

from langchain_community.embeddings import (
    BedrockEmbeddings,
    HuggingFaceEmbeddings,
    OllamaEmbeddings,
)
from langchain_core.embeddings import Embeddings

from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CACHE_SIZE = 1
MAX_RETRIES = 3


class EmbeddingType(Enum):
    """Supported embedding types."""

    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"


class EmbeddingError(Exception):
    """Base class for embedding-related errors."""

    pass


class EmbeddingProvider:
    """Factory class for creating embedding functions with fallback support."""

    def __init__(self):
        """Initialize the embedding provider with fallback options."""
        self._providers: Dict[EmbeddingType, Callable[[], Embeddings]] = {
            EmbeddingType.OLLAMA: self.create_ollama_embeddings,
            EmbeddingType.BEDROCK: self.create_bedrock_embeddings,
            EmbeddingType.HUGGINGFACE: self.create_huggingface_embeddings,
        }
        self._fallback_order = [
            EmbeddingType.OLLAMA,
            EmbeddingType.HUGGINGFACE,
            EmbeddingType.BEDROCK,
        ]

    @staticmethod
    def create_ollama_embeddings() -> OllamaEmbeddings:
        """Create Ollama embeddings with retry mechanism.

        Returns:
            OllamaEmbeddings instance

        Raises:
            EmbeddingError: If embedding creation fails
        """
        try:
            return OllamaEmbeddings(
                model=config.EMBEDDING_MODEL,
                model_kwargs={
                    "device": (
                        "cuda"
                        if os.getenv("USE_GPU", "false").lower() == "true"
                        else "cpu"
                    )
                },
            )
        except Exception as e:
            logger.error(f"Failed to create Ollama embeddings: {str(e)}")
            raise EmbeddingError(f"Ollama embeddings creation failed: {str(e)}")

    @staticmethod
    def create_bedrock_embeddings() -> BedrockEmbeddings:
        """Create AWS Bedrock embeddings.

        Returns:
            BedrockEmbeddings instance

        Raises:
            EmbeddingError: If embedding creation fails
        """
        try:
            return BedrockEmbeddings(
                credentials_profile_name=os.getenv("AWS_PROFILE", "default"),
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                model_id=os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v1"),
            )
        except Exception as e:
            logger.error(f"Failed to create Bedrock embeddings: {str(e)}")
            raise EmbeddingError(f"Bedrock embeddings creation failed: {str(e)}")

    @staticmethod
    def create_huggingface_embeddings() -> HuggingFaceEmbeddings:
        """Create HuggingFace embeddings.

        Returns:
            HuggingFaceEmbeddings instance

        Raises:
            EmbeddingError: If embedding creation fails
        """
        try:
            return HuggingFaceEmbeddings(
                model_name=os.getenv(
                    "HF_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"
                ),
                model_kwargs={
                    "device": (
                        "cuda"
                        if os.getenv("USE_GPU", "false").lower() == "true"
                        else "cpu"
                    )
                },
            )
        except Exception as e:
            logger.error(f"Failed to create HuggingFace embeddings: {str(e)}")
            raise EmbeddingError(f"HuggingFace embeddings creation failed: {str(e)}")

    def get_embedding_with_fallback(self) -> Embeddings:
        """Get embedding function with automatic fallback.

        Returns:
            Embeddings instance from the first successful provider

        Raises:
            EmbeddingError: If all providers fail
        """
        errors = []

        # Try each provider in fallback order
        for provider_type in self._fallback_order:
            try:
                provider_func = self._providers[provider_type]
                embeddings = provider_func()
                logger.info(
                    f"Successfully initialized {provider_type.value} embeddings"
                )
                return embeddings
            except Exception as e:
                errors.append(f"{provider_type.value}: {str(e)}")
                continue

        # If all providers fail, raise error with details
        error_msg = "All embedding providers failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise EmbeddingError(error_msg)


@lru_cache(maxsize=CACHE_SIZE)
def get_embedding_function() -> Embeddings:
    """Get the configured embedding function with caching.

    This function caches the embedding provider to avoid repeated initialization.
    The cache is limited to one instance to prevent memory issues.

    Returns:
        Cached embedding function instance

    Raises:
        EmbeddingError: If no embedding provider can be initialized
    """
    try:
        provider = EmbeddingProvider()
        return provider.get_embedding_with_fallback()
    except Exception as e:
        logger.error(f"Failed to get embedding function: {str(e)}")
        raise EmbeddingError(f"Failed to initialize any embedding provider: {str(e)}")
