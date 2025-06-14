import logging
import re

from climatefact.workflows.contradiction_detection.types import ContradictionDetectionState

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> list[str]:
    """
    Split paragraph text into a list of sentences.
    """
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


# Placeholder: use nltk or spacy for actual sentence segmentation
def segment_sentences(state: ContradictionDetectionState) -> ContradictionDetectionState:
    """
    Segments the input text into sentences.
    """
    logger.info("---SEGMENTING TEXT---")
    input_text = state["input_text"]
    sentences = split_into_sentences(input_text)
    logger.info(f"Found {len(sentences)} sentences.")

    # Ensure sentences is always a list, even if input_text is empty or has no periods
    if not sentences and input_text:  # Handle case where input_text might not have periods but is not empty
        sentences = [input_text.strip()]

    updated_state = state.copy()
    updated_state["queries"] = sentences  # Populate 'queries' directly
    return updated_state
