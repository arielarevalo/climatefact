"""Shared utilities for embeddings and vector search."""

import logging
import os
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

logger = logging.getLogger(__name__)

# Load environment variables once at module level
load_dotenv()

# Initialize Azure OpenAI embeddings client (singleton pattern)
_embeddings_client: AzureOpenAIEmbeddings | None = None


def validate_azure_config() -> dict[str, str | None]:
    """Validate and return Azure OpenAI configuration."""
    config = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    }

    missing_vars = [key for key, value in config.items() if not value]
    if missing_vars:
        logger.error(
            f"Missing required Azure OpenAI environment variables: {', '.join(var.upper() for var in missing_vars)}"
        )
        return {}

    # Validate endpoint format
    if config["endpoint"] and not config["endpoint"].startswith("https://"):
        logger.error(f"Invalid Azure OpenAI endpoint format: {config['endpoint']}")
        return {}

    return config


def get_embeddings_client() -> AzureOpenAIEmbeddings | None:
    """Initialize and return Azure OpenAI embeddings client (singleton)."""
    global _embeddings_client

    if _embeddings_client is not None:
        return _embeddings_client

    config = validate_azure_config()
    if not config:
        logger.error("Invalid Azure OpenAI configuration")
        return None

    try:
        _embeddings_client = AzureOpenAIEmbeddings(
            azure_endpoint=config["endpoint"],
            azure_deployment=config["deployment"],
            api_version=config["api_version"],
        )
        logger.info(f"Successfully initialized Azure OpenAI embeddings client with deployment: {config['deployment']}")
        return _embeddings_client

    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI embeddings client: {e}")
        return None


def get_query_embedding(query: str) -> list[float]:
    """Get embedding for query using LangChain Azure OpenAI embeddings."""
    if not query.strip():
        logger.warning("Empty query provided for embedding")
        return [0.0] * 1536

    embeddings_client = get_embeddings_client()

    if not embeddings_client:
        logger.warning(f"No embeddings client available, using dummy embedding for query: '{query[:50]}...'")
        return [0.0] * 1536

    try:
        embedding = embeddings_client.embed_query(query)
        logger.debug(f"Generated embedding for query: '{query[:50]}...'")
        return embedding

    except Exception as e:
        logger.error(f"Error getting embedding for query '{query[:50]}...': {e}")
        logger.warning("Falling back to dummy embedding")
        return [0.0] * 1536


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1_array = np.array(vec1)
    vec2_array = np.array(vec2)

    dot_product = np.dot(vec1_array, vec2_array)
    norm1 = np.linalg.norm(vec1_array)
    norm2 = np.linalg.norm(vec2_array)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def vector_search(query: str, passages: list[dict[str, Any]], *, top_k: int = 3) -> list[dict[str, Any]]:
    """Vector-based retrieval using cosine similarity with embeddings."""
    query_embedding = get_query_embedding(query)
    scored_passages = []

    for passage in passages:
        passage_embedding = passage.get("embedding")
        if not passage_embedding:
            logger.warning(f"No embedding found for passage: {passage.get('text', '')[:50]}...")
            continue

        # Calculate similarity
        similarity = cosine_similarity(query_embedding, passage_embedding)
        if similarity > 0:  # Only consider passages with positive similarity
            scored_passages.append((similarity, passage))

    # Sort by similarity and return top_k
    scored_passages.sort(key=lambda x: x[0], reverse=True)
    return [passage for _, passage in scored_passages[:top_k]]
