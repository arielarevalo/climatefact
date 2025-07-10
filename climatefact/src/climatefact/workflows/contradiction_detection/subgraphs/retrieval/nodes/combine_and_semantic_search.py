from climatefact.workflows.contradiction_detection.subgraphs.retrieval.types import RetrievalState
from langchain_core.runnables import RunnableConfig
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import logging
import os

logger = logging.getLogger(__name__)
load_dotenv()

# Reuse the same singleton pattern from semantic search node
_embeddings_client: Optional[AzureOpenAIEmbeddings] = None

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
        logger.error(f"Missing required Azure OpenAI environment variables: {', '.join(var.upper() for var in missing_vars)}")
        return {}

    if not config["endpoint"].startswith("https://"):
        logger.error(f"Invalid Azure OpenAI endpoint format: {config['endpoint']}")
        return {}

    return config

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
        logger.info(f"Successfully initialized Azure OpenAI embeddings client")
        return _embeddings_client

    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI embeddings client: {e}")
        return None

def get_query_embedding(query: str) -> List[float]:
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

def vector_search_on_passages(query: str, passages: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Vector-based search on a specific set of passages."""
    if not passages:
        return []
        
    query_embedding = get_query_embedding(query)
    scored_passages = []

    for passage in passages:
        passage_embedding = passage.get("embedding")
        if not passage_embedding:
            logger.debug(f"No embedding found for passage: {passage.get('text', '')[:50]}...")
            continue

        similarity = cosine_similarity(query_embedding, passage_embedding)
        if similarity > 0:
            scored_passages.append((similarity, passage))

    # Sort by similarity and return top_k
    scored_passages.sort(key=lambda x: x[0], reverse=True)
    return [passage for _, passage in scored_passages[:top_k]]

def deduplicate_passages(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate passages based on their ID, preserving order."""
    seen_ids = set()
    deduplicated = []
    
    for passage in passages:
        passage_id = passage.get('id', '')
        if passage_id and passage_id not in seen_ids:
            seen_ids.add(passage_id)
            deduplicated.append(passage)
        elif not passage_id:
            # If no ID, check text similarity to avoid duplicates
            passage_text = passage.get('text', '')
            if passage_text and not any(p.get('text', '') == passage_text for p in deduplicated):
                deduplicated.append(passage)
    
    return deduplicated

def combine_and_semantic_search_node(state: RetrievalState, config: RunnableConfig) -> RetrievalState:
    """
    Combines passages from regex and NER retrieval, applies semantic search.
    Results are stored in hybrid_retrieved_data.
    """
    queries = state.get("queries", [])
    regex_data = state.get("regex_retrieved_data", [])
    ner_data = state.get("ner_retrieved_data", [])
    
    logger.info(f"---COMBINING REGEX+NER AND APPLYING SEMANTIC SEARCH FOR {len(queries)} QUERIES---")
    
    hybrid_retrieved_data: List[List[Dict[str, Any]]] = []
    
    top_k = config["configurable"].get("hybrid_retrieval_top_k", 5)
    logger.info(f"Using top_k={top_k} for hybrid retrieval")
    for i, query in enumerate(queries):
        logger.info(f"Processing query {i}: '{query}'")
        
        # Get passages from both methods for this query
        regex_passages = regex_data[i] if i < len(regex_data) else []
        ner_passages = ner_data[i] if i < len(ner_data) else []
        
        logger.info(f"Regex passages: {len(regex_passages)}, NER passages: {len(ner_passages)}")
        
        # Combine and deduplicate passages
        combined_passages = deduplicate_passages(regex_passages + ner_passages)
        logger.info(f"Combined unique passages: {len(combined_passages)}")
        
        if combined_passages:
            # Apply semantic search on combined passages
            top_passages = vector_search_on_passages(query, combined_passages, top_k=3)
            logger.info(f"Top semantic matches: {len(top_passages)}")
            
            # Clean up passages (remove embeddings)
            cleaned_passages = []
            for passage in top_passages:
                cleaned_passage = passage.copy()
                if "embedding" in cleaned_passage:
                    del cleaned_passage["embedding"]
                cleaned_passages.append(cleaned_passage)
            
            hybrid_retrieved_data.append(cleaned_passages)
        else:
            logger.info("No passages to process for this query")
            hybrid_retrieved_data.append([])
    
    logger.info(f"Finished hybrid retrieval for all {len(queries)} queries")
    
    return {"hybrid_retrieved_data": hybrid_retrieved_data}
