import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, NamedTuple

from dotenv import load_dotenv
from langchain_community.llms.azureml_endpoint import (
    AzureMLEndpointApiType,
    AzureMLOnlineEndpoint,
    ContentFormatterBase,
)
from langchain_core.outputs import Generation
from pydantic import SecretStr

from climatefact.workflows.contradiction_detection.types import (
    ContradictionDetectionState,
    ContradictionEvidence,
    ContradictionResult,
)

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables once at module level
load_dotenv()


@dataclass
class NLITask:
    """Represents a single NLI inference task."""

    sentence_idx: int
    passage_idx: int
    sentence: str
    passage_text: str
    passage_source: str


class NLIResult(NamedTuple):
    """Result of NLI inference."""

    sentence_idx: int
    passage_idx: int
    sentence: str
    passage_text: str
    passage_source: str
    label: str


class NLIContentFormatter(ContentFormatterBase):
    """Custom content formatter for NLI (Natural Language Inference) models."""

    content_type = "application/json"
    accepts = "application/json"

    def format_request_payload(
        self, prompt: str, model_kwargs: dict, api_type: AzureMLEndpointApiType = AzureMLEndpointApiType.dedicated
    ) -> bytes:
        """Format the request payload for NLI model."""
        # For NLI models, the prompt should contain the premise and hypothesis
        input_str = json.dumps(
            {"inputs": prompt, "parameters": model_kwargs, "options": {"use_cache": False, "wait_for_model": True}}
        )
        return str.encode(input_str)

    def format_response_payload(
        self, output: bytes, api_type: AzureMLEndpointApiType = AzureMLEndpointApiType.dedicated
    ) -> Generation:
        """Format the response payload from NLI model."""
        try:
            response_json = json.loads(output)
            # Handle different response formats
            if isinstance(response_json, list) and len(response_json) > 0:
                if isinstance(response_json[0], dict):
                    label = response_json[0].get("label", "NEUTRAL").lower()
                elif isinstance(response_json[0], list) and len(response_json[0]) > 0:
                    label = response_json[0][0].get("label", "NEUTRAL").lower()
                else:
                    label = "neutral"
            elif isinstance(response_json, dict):
                label = response_json.get("label", "NEUTRAL").lower()
            else:
                label = "neutral"

            return Generation(text=label)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing NLI response: {e}")
            return Generation(text="neutral")


def validate_azure_nli_config() -> dict[str, str]:
    """Validate and return Azure ML NLI configuration."""
    endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
    api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL")

    if not endpoint or not api_key:
        missing_vars = []
        if not endpoint:
            missing_vars.append("AZURE_INFERENCE_ENDPOINT")
        if not api_key:
            missing_vars.append("AZURE_INFERENCE_CREDENTIAL")
        logger.error(f"Missing required Azure ML NLI environment variables: {', '.join(missing_vars)}")
        return {}

    # Validate endpoint format
    if not endpoint.startswith("https://"):
        logger.error(f"Invalid Azure ML NLI endpoint format: {endpoint}")
        return {}

    return {
        "endpoint": endpoint,
        "api_key": api_key,
    }


# Initialize Azure ML NLI client (singleton pattern)
_nli_client: AzureMLOnlineEndpoint | None = None


def get_nli_client() -> AzureMLOnlineEndpoint | None:
    """Initialize and return Azure ML NLI client (singleton)."""
    global _nli_client

    if _nli_client is not None:
        return _nli_client

    config = validate_azure_nli_config()
    if not config:
        logger.error("Invalid Azure ML NLI configuration")
        return None

    try:
        content_formatter = NLIContentFormatter()

        _nli_client = AzureMLOnlineEndpoint(
            endpoint_url=config["endpoint"],
            endpoint_api_key=SecretStr(config["api_key"]),
            endpoint_api_type=AzureMLEndpointApiType.dedicated,
            content_formatter=content_formatter,
        )
        logger.info(f"Successfully initialized Azure ML NLI client with endpoint: {config['endpoint']}")
        return _nli_client

    except Exception as e:
        logger.error(f"Failed to initialize Azure ML NLI client: {e}")
        return None


def _call_nli_model_sync(client: AzureMLOnlineEndpoint, premise: str, hypothesis: str) -> str:
    """Synchronous NLI model call - used by concurrent executor."""
    if not premise.strip() or not hypothesis.strip():
        logger.warning("Empty premise or hypothesis provided for NLI")
        return "neutral"

    try:
        # Format input for NLI model with proper [CLS] and [SEP] tokens
        nli_input = f"[CLS] {premise} [SEP] {hypothesis} [SEP]"
        result = client.invoke(nli_input)

        # The result should be a Generation object from our custom formatter
        if isinstance(result, Generation):
            label = result.text.lower()
        elif isinstance(result, str):
            label = result.lower()
        else:
            label = "neutral"

        logger.debug(f"NLI result for premise '{premise[:30]}...' and hypothesis '{hypothesis[:30]}...': {label}")
        return label

    except Exception as e:
        logger.error(
            f"Error calling NLI model for premise '{premise[:30]}...' and hypothesis '{hypothesis[:30]}...': {e}"
        )
        logger.warning("Falling back to neutral label")
        return "neutral"


def _process_nli_task(client: AzureMLOnlineEndpoint, task: NLITask) -> NLIResult:
    """Process a single NLI task and return the result."""
    label = _call_nli_model_sync(client, task.passage_text, task.sentence)
    return NLIResult(
        sentence_idx=task.sentence_idx,
        passage_idx=task.passage_idx,
        sentence=task.sentence,
        passage_text=task.passage_text,
        passage_source=task.passage_source,
        label=label,
    )


def _prepare_nli_tasks(
    queries: list[str], retrieved_passages_for_sentences: list[list[dict[str, Any]]]
) -> list[NLITask]:
    """Prepare all NLI tasks for concurrent processing."""
    tasks = []

    for sentence_idx, sentence in enumerate(queries):
        if sentence_idx >= len(retrieved_passages_for_sentences):
            logger.warning(f"Skipping sentence '{sentence}' due to missing retrieved passages set.")
            continue

        retrieved_passages = retrieved_passages_for_sentences[sentence_idx]

        for passage_idx, passage in enumerate(retrieved_passages):
            passage_text = passage.get("text", "")
            passage_source = passage.get("source", "Unknown source")

            if passage_text.strip():  # Only add non-empty passages
                tasks.append(
                    NLITask(
                        sentence_idx=sentence_idx,
                        passage_idx=passage_idx,
                        sentence=sentence,
                        passage_text=passage_text,
                        passage_source=passage_source,
                    )
                )

    return tasks


def _process_nli_tasks_concurrently(
    client: AzureMLOnlineEndpoint, tasks: list[NLITask], max_workers: int = 10
) -> list[NLIResult]:
    """Process all NLI tasks concurrently using ThreadPoolExecutor."""
    if not tasks:
        return []

    logger.info(f"Processing {len(tasks)} NLI tasks concurrently with {max_workers} workers")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(_process_nli_task, client, task): task for task in tasks}

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                logger.debug(
                    f"Completed NLI task for sentence {result.sentence_idx}, "
                    f"passage {result.passage_idx}: {result.label}"
                )
            except Exception as e:
                logger.error(
                    f"Error processing NLI task for sentence {task.sentence_idx}, passage {task.passage_idx}: {e}"
                )
                # Add a neutral result for failed tasks
                results.append(
                    NLIResult(
                        sentence_idx=task.sentence_idx,
                        passage_idx=task.passage_idx,
                        sentence=task.sentence,
                        passage_text=task.passage_text,
                        passage_source=task.passage_source,
                        label="neutral",
                    )
                )

    logger.info(f"Completed processing {len(results)} NLI tasks")
    return results


def _organize_nli_results(nli_results: list[NLIResult], queries: list[str]) -> list[ContradictionResult]:
    """Organize NLI results into ContradictionResult objects."""
    # Group results by sentence index
    results_by_sentence = {}
    for result in nli_results:
        if result.sentence_idx not in results_by_sentence:
            results_by_sentence[result.sentence_idx] = []
        results_by_sentence[result.sentence_idx].append(result)

    contradiction_results = []

    for sentence_idx, sentence in enumerate(queries):
        sentence_contradictions = []
        sentence_results = results_by_sentence.get(sentence_idx, [])

        for result in sentence_results:
            if result.label == "contradiction":
                sentence_contradictions.append(
                    ContradictionEvidence(contradictory_passage=result.passage_text, source=result.passage_source)
                )
                logger.info(f"Found contradiction: '{sentence}' vs '{result.passage_text[:50]}...'")

        contradiction_results.append(
            ContradictionResult(
                sentence=sentence,
                contradictions=sentence_contradictions,
                has_contradictions=len(sentence_contradictions) > 0,
            )
        )

    return contradiction_results


def detect_contradictions(state: ContradictionDetectionState) -> ContradictionDetectionState:
    """
    Detects contradictions between sentences and their retrieved passages.
    Reads 'queries' and 'retrieved_data_for_queries' from the state.
    Uses concurrent processing for all NLI calls.
    """
    logger.info("---DETECTING CONTRADICTIONS WITH CONCURRENT PROCESSING---")

    # Get data from state with proper defaults
    queries = state.get("queries", [])
    retrieved_passages_for_sentences = state.get("retrieved_data_for_queries", [])

    # Handle None case
    if retrieved_passages_for_sentences is None:
        retrieved_passages_for_sentences = []

    logger.info(f"Received {len(queries)} queries for contradiction detection")
    logger.info(f"Received {len(retrieved_passages_for_sentences)} passage sets for contradiction detection")

    # Log the structure of retrieved data
    for i, passage_set in enumerate(retrieved_passages_for_sentences):
        logger.info(f"Passage set {i}: {len(passage_set) if passage_set else 0} passages")
        if passage_set:
            for j, passage in enumerate(passage_set[:2]):  # Log first 2 passages
                logger.debug(
                    f"  Passage {j}: {passage.get('text', '')[:100]}... from {passage.get('source', 'Unknown')}"
                )

    # Validate input
    if not queries:
        logger.warning("No queries provided for contradiction detection")
        updated_state = state.copy()
        updated_state["contradiction_results"] = []
        return updated_state

    if len(queries) != len(retrieved_passages_for_sentences):
        logger.warning(
            f"Mismatch between number of queries ({len(queries)}) and "
            f"retrieved passage sets ({len(retrieved_passages_for_sentences)})."
        )

    # Initialize NLI client
    nli_client = get_nli_client()
    if not nli_client:
        logger.error("Failed to initialize NLI client, returning empty results")
        updated_state = state.copy()
        updated_state["contradiction_results"] = [
            ContradictionResult(sentence=sentence, contradictions=[], has_contradictions=False) for sentence in queries
        ]
        return updated_state

    # Prepare all NLI tasks
    nli_tasks = _prepare_nli_tasks(queries, retrieved_passages_for_sentences)
    logger.info(f"Prepared {len(nli_tasks)} NLI tasks for concurrent processing")

    if not nli_tasks:
        logger.info("No NLI tasks to process")
        updated_state = state.copy()
        updated_state["contradiction_results"] = [
            ContradictionResult(sentence=sentence, contradictions=[], has_contradictions=False) for sentence in queries
        ]
        return updated_state

    # Process all NLI tasks concurrently
    nli_results = _process_nli_tasks_concurrently(nli_client, nli_tasks)

    # Organize results into ContradictionResult objects
    contradiction_results = _organize_nli_results(nli_results, queries)

    # Update state
    updated_state = state.copy()
    updated_state["contradiction_results"] = contradiction_results

    # Log summary
    total_contradictions = sum(len(result["contradictions"]) for result in contradiction_results)
    sentences_with_contradictions = sum(1 for result in contradiction_results if result["has_contradictions"])
    logger.info(
        f"Found {total_contradictions} total contradictions across "
        f"{sentences_with_contradictions} sentences using concurrent processing."
    )

    return updated_state
