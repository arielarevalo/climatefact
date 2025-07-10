from climatefact.workflows.contradiction_detection.subgraphs.retrieval.types import RetrievalState
from langchain_core.runnables import RunnableConfig
from typing import List, Dict, Any, Optional
import json
import os
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables once at module level
load_dotenv()


def validate_azure_config() -> Dict[str, Optional[str]]:
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
    if not config["endpoint"].startswith("https://"):
        logger.error(f"Invalid Azure OpenAI endpoint format: {config['endpoint']}")
        return {}

    return config


# Initialize Azure OpenAI embeddings client (singleton pattern)
_embeddings_client: Optional[AzureOpenAIEmbeddings] = None


def get_embeddings_client() -> Optional[AzureOpenAIEmbeddings]:
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
            api_version=config["api_version"]
        )
        logger.info(
            f"Successfully initialized Azure OpenAI embeddings client with deployment: {config['deployment']}"
        )
        return _embeddings_client

    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI embeddings client: {e}")
        return None


def load_passages_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load passages from a JSONL file."""
    passages = []
    if not os.path.exists(jsonl_path):
        logger.warning(f"JSONL file not found at {jsonl_path}")
        return passages

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                passage = json.loads(line.strip())
                passages.append(passage)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

    logger.info(f"Loaded {len(passages)} passages from {jsonl_path}")
    return passages


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def get_query_embedding(query: str) -> List[float]:
    """Get embedding for query using LangChain Azure OpenAI embeddings."""
    if not query.strip():
        logger.warning("Empty query provided for embedding")
        return [0.0] * 1536

    embeddings_client = get_embeddings_client()

    if not embeddings_client:
        logger.warning(
            f"No embeddings client available, using dummy embedding for query: '{query[:50]}...'"
        )
        return [0.0] * 1536

    try:
        embedding = embeddings_client.embed_query(query)
        logger.debug(f"Generated embedding for query: '{query[:50]}...'")
        return embedding

    except Exception as e:
        logger.error(f"Error getting embedding for query '{query[:50]}...': {e}")
        logger.warning("Falling back to dummy embedding")
        return [0.0] * 1536


def vector_search(
    query: str, passages: List[Dict[str, Any]], top_k: int = 3
) -> List[Dict[str, Any]]:
    """Vector-based retrieval using cosine similarity with embeddings."""
    query_embedding = get_query_embedding(query)
    scored_passages = []

    for passage in passages:
        passage_embedding = passage.get("embedding")
        if not passage_embedding:
            logger.warning(
                f"No embedding found for passage: {passage.get('text', '')[:50]}..."
            )
            continue

        # Calculate similarity
        similarity = cosine_similarity(query_embedding, passage_embedding)
        if similarity > 0:  # Only consider passages with positive similarity
            scored_passages.append((similarity, passage))

    # Sort by similarity and return top_k
    scored_passages.sort(key=lambda x: x[0], reverse=True)
    return [passage for _, passage in scored_passages[:top_k]]


def retrieve_by_semantic_search_node(state: RetrievalState, config: RunnableConfig) -> RetrievalState:
    """
    Retrieves relevant passages for a list of given queries from a JSONL file.
    Uses vector similarity if embeddings are available, falls back to text search.
    Gets 'passages_jsonl_path' from configuration.
    """
    queries = state.get("queries", [])
    jsonl_path = config["configurable"]["passages_jsonl_path"]

    if not jsonl_path:
        logger.error("No 'passages_jsonl_path' provided in configuration")
        return {"semantic_retrieved_data": []}

    logger.info(
        f"---RETRIEVING PASSAGES FOR {len(queries)} QUERIES FROM {jsonl_path}---"
    )

    # Load all passages from JSONL file
    all_passages = load_passages_from_jsonl(jsonl_path)
    if not all_passages:
        logger.warning("No passages loaded, returning empty results")
        return {"semantic_retrieved_data": []}

    all_retrieved_passages: List[List[Dict[str, Any]]] = []

    top_k = config["configurable"].get("semantic_search_top_k", 5)
    logger.info(f"Using top_k={top_k} for semantic retrieval")
    for query_text in queries:
        logger.info(f"Retrieving for query: '{query_text}'")

        logger.info("Using vector search with embeddings")
        passages_for_query = vector_search(query_text, all_passages, top_k=top_k)

        # Clean up passages for output (remove embeddings to save memory)
        cleaned_passages = []
        for passage in passages_for_query:
            cleaned_passage = passage.copy()
            if "embedding" in cleaned_passage:
                del cleaned_passage["embedding"]  # Remove large embedding vector
            cleaned_passages.append(cleaned_passage)

        all_retrieved_passages.append(cleaned_passages)
        logger.info(
            f"Retrieved {len(passages_for_query)} passages for query '{query_text}'."
        )

    # Add detailed logging to verify state content
    logger.info(f"Finished semantic retrieval for all {len(queries)} queries.")
    logger.info(f"Total retrieved passage sets: {len(all_retrieved_passages)}")
    for i, passage_set in enumerate(all_retrieved_passages):
        logger.info(f"Query {i}: {len(passage_set)} passages retrieved")
        for j, passage in enumerate(passage_set):
            logger.debug(f"  Passage {j}: {passage.get('text', '')[:100]}...")
    
    return {"semantic_retrieved_data": all_retrieved_passages}
