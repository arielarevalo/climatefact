"""
Tests for concurrent contradiction detection functionality.
"""

from typing import cast
from unittest.mock import Mock, patch

import pytest

from climatefact.workflows.contradiction_detection.nodes.detect_contradictions import (
    NLIContentFormatter,
    NLIResult,
    NLITask,
    _call_nli_model_sync,
    _organize_nli_results,
    _prepare_nli_tasks,
    _process_nli_task,
    _process_nli_tasks_concurrently,
    detect_contradictions,
    validate_azure_nli_config,
)
from climatefact.workflows.contradiction_detection.types import (
    ContradictionDetectionState,
)


class TestNLIContentFormatter:
    """Test the NLI content formatter."""

    def test_format_request_payload(self):
        formatter = NLIContentFormatter()
        payload = formatter.format_request_payload("test prompt", {})

        assert isinstance(payload, bytes)
        # Should be valid JSON
        import json

        payload_dict = json.loads(payload.decode())
        assert "inputs" in payload_dict
        assert payload_dict["inputs"] == "test prompt"

    def test_format_response_payload_list_dict(self):
        formatter = NLIContentFormatter()
        response_data = [{"label": "CONTRADICTION"}]
        response_bytes = str.encode(str(response_data).replace("'", '"'))

        generation = formatter.format_response_payload(response_bytes)
        assert generation.text == "contradiction"

    def test_format_response_payload_nested_list(self):
        formatter = NLIContentFormatter()
        response_data = [[{"label": "ENTAILMENT"}]]
        response_bytes = str.encode(str(response_data).replace("'", '"'))

        generation = formatter.format_response_payload(response_bytes)
        assert generation.text == "entailment"

    def test_format_response_payload_dict(self):
        formatter = NLIContentFormatter()
        response_data = {"label": "NEUTRAL"}
        response_bytes = str.encode(str(response_data).replace("'", '"'))

        generation = formatter.format_response_payload(response_bytes)
        assert generation.text == "neutral"

    def test_format_response_payload_error(self):
        formatter = NLIContentFormatter()
        # Invalid JSON
        response_bytes = b"invalid json"

        generation = formatter.format_response_payload(response_bytes)
        assert generation.text == "neutral"


class TestConfigValidation:
    """Test Azure NLI configuration validation."""

    @patch.dict(
        "os.environ",
        {"AZURE_INFERENCE_ENDPOINT": "https://test.endpoint.com", "AZURE_INFERENCE_CREDENTIAL": "test-key"},
    )
    def test_validate_azure_nli_config_valid(self):
        config = validate_azure_nli_config()
        assert config["endpoint"] == "https://test.endpoint.com"
        assert config["api_key"] == "test-key"

    @patch.dict("os.environ", {}, clear=True)
    def test_validate_azure_nli_config_missing_vars(self):
        config = validate_azure_nli_config()
        assert config == {}

    @patch.dict(
        "os.environ",
        {"AZURE_INFERENCE_ENDPOINT": "http://invalid.endpoint.com", "AZURE_INFERENCE_CREDENTIAL": "test-key"},
    )
    def test_validate_azure_nli_config_invalid_endpoint(self):
        config = validate_azure_nli_config()
        assert config == {}


class TestNLIDataStructures:
    """Test NLI data structures."""

    def test_nli_task_creation(self):
        task = NLITask(
            sentence_idx=0,
            passage_idx=1,
            sentence="Test sentence",
            passage_text="Test passage",
            passage_source="Test source",
        )
        assert task.sentence_idx == 0
        assert task.passage_idx == 1
        assert task.sentence == "Test sentence"

    def test_nli_result_creation(self):
        result = NLIResult(
            sentence_idx=0,
            passage_idx=1,
            sentence="Test sentence",
            passage_text="Test passage",
            passage_source="Test source",
            label="contradiction",
        )
        assert result.label == "contradiction"
        assert result.sentence_idx == 0


class TestNLIProcessing:
    """Test NLI processing functions."""

    def test_prepare_nli_tasks(self):
        queries = ["sentence 1", "sentence 2"]
        passages = [
            [{"text": "passage 1", "source": "source 1"}],
            [{"text": "passage 2", "source": "source 2"}, {"text": "passage 3", "source": "source 3"}],
        ]

        tasks = _prepare_nli_tasks(queries, passages)

        assert len(tasks) == 3
        assert tasks[0].sentence == "sentence 1"
        assert tasks[0].passage_text == "passage 1"
        assert tasks[1].sentence == "sentence 2"
        assert tasks[2].sentence == "sentence 2"

    def test_prepare_nli_tasks_empty_passages(self):
        queries = ["sentence 1"]
        passages = [[{"text": "", "source": "source 1"}]]

        tasks = _prepare_nli_tasks(queries, passages)

        assert len(tasks) == 0  # Empty passages should be filtered out

    def test_prepare_nli_tasks_mismatched_lengths(self):
        queries = ["sentence 1", "sentence 2"]
        passages = [[{"text": "passage 1", "source": "source 1"}]]

        tasks = _prepare_nli_tasks(queries, passages)

        assert len(tasks) == 1  # Only first sentence processed

    @patch("climatefact.workflows.contradiction_detection.nodes.detect_contradictions._call_nli_model_sync")
    def test_process_nli_task(self, mock_nli_call):
        mock_nli_call.return_value = "contradiction"

        client = Mock()
        task = NLITask(0, 0, "sentence", "passage", "source")

        result = _process_nli_task(client, task)

        assert result.label == "contradiction"
        assert result.sentence_idx == 0
        mock_nli_call.assert_called_once_with(client, "passage", "sentence")

    def test_call_nli_model_sync_empty_input(self):
        client = Mock()
        result = _call_nli_model_sync(client, "", "hypothesis")
        assert result == "neutral"

        result = _call_nli_model_sync(client, "premise", "")
        assert result == "neutral"

    @patch("climatefact.workflows.contradiction_detection.nodes.detect_contradictions._process_nli_task")
    def test_process_nli_tasks_concurrently(self, mock_process_task):
        # Mock the task processing
        def mock_task_processor(client, task):
            return NLIResult(
                sentence_idx=task.sentence_idx,
                passage_idx=task.passage_idx,
                sentence=task.sentence,
                passage_text=task.passage_text,
                passage_source=task.passage_source,
                label="neutral",
            )

        mock_process_task.side_effect = mock_task_processor

        client = Mock()
        tasks = [
            NLITask(0, 0, "sentence 1", "passage 1", "source 1"),
            NLITask(1, 0, "sentence 2", "passage 2", "source 2"),
        ]

        results = _process_nli_tasks_concurrently(client, tasks, max_workers=2)

        assert len(results) == 2
        assert all(result.label == "neutral" for result in results)

    def test_organize_nli_results(self):
        queries = ["sentence 1", "sentence 2"]
        nli_results = [
            NLIResult(0, 0, "sentence 1", "passage 1", "source 1", "contradiction"),
            NLIResult(0, 1, "sentence 1", "passage 2", "source 2", "neutral"),
            NLIResult(1, 0, "sentence 2", "passage 3", "source 3", "contradiction"),
        ]

        contradiction_results = _organize_nli_results(nli_results, queries)

        assert len(contradiction_results) == 2
        assert contradiction_results[0]["sentence"] == "sentence 1"
        assert len(contradiction_results[0]["contradictions"]) == 1
        assert contradiction_results[0]["has_contradictions"] is True

        assert contradiction_results[1]["sentence"] == "sentence 2"
        assert len(contradiction_results[1]["contradictions"]) == 1
        assert contradiction_results[1]["has_contradictions"] is True


class TestDetectContradictions:
    """Test the main detect_contradictions function."""

    @patch("climatefact.workflows.contradiction_detection.nodes.detect_contradictions.get_nli_client")
    @patch("climatefact.workflows.contradiction_detection.nodes.detect_contradictions._process_nli_tasks_concurrently")
    def test_detect_contradictions_success(self, mock_concurrent_process, mock_get_client):
        # Setup mocks
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_concurrent_process.return_value = [
            NLIResult(0, 0, "test sentence", "test passage", "test source", "contradiction")
        ]

        # Prepare state
        state = cast(
            ContradictionDetectionState,
            {
                "input_text": "test input",
                "queries": ["test sentence"],
                "regex_retrieved_data": None,
                "ner_retrieved_data": None,
                "semantic_retrieved_data": None,
                "hybrid_retrieved_data": None,
                "retrieved_data_for_queries": [[{"text": "test passage", "source": "test source"}]],
                "contradiction_results": None,
                "report": None,
            },
        )

        # Execute
        updated_state = detect_contradictions(state)

        # Verify
        assert "contradiction_results" in updated_state
        results = updated_state["contradiction_results"]
        assert results is not None
        assert len(results) == 1
        assert results[0]["sentence"] == "test sentence"
        assert results[0]["has_contradictions"] is True
        assert len(results[0]["contradictions"]) == 1

    @patch("climatefact.workflows.contradiction_detection.nodes.detect_contradictions.get_nli_client")
    def test_detect_contradictions_no_client(self, mock_get_client):
        mock_get_client.return_value = None

        state = cast(
            ContradictionDetectionState,
            {
                "input_text": "test input",
                "queries": ["test sentence"],
                "regex_retrieved_data": None,
                "ner_retrieved_data": None,
                "semantic_retrieved_data": None,
                "hybrid_retrieved_data": None,
                "retrieved_data_for_queries": [[{"text": "test passage", "source": "test source"}]],
                "contradiction_results": None,
                "report": None,
            },
        )

        updated_state = detect_contradictions(state)

        assert "contradiction_results" in updated_state
        results = updated_state["contradiction_results"]
        assert results is not None
        assert len(results) == 1
        assert results[0]["has_contradictions"] is False

    def test_detect_contradictions_empty_queries(self):
        state = cast(
            ContradictionDetectionState,
            {
                "input_text": "test input",
                "queries": [],
                "regex_retrieved_data": None,
                "ner_retrieved_data": None,
                "semantic_retrieved_data": None,
                "hybrid_retrieved_data": None,
                "retrieved_data_for_queries": [],
                "contradiction_results": None,
                "report": None,
            },
        )

        updated_state = detect_contradictions(state)

        assert "contradiction_results" in updated_state
        results = updated_state["contradiction_results"]
        assert results is not None
        assert len(results) == 0

    def test_detect_contradictions_none_passages(self):
        state = cast(
            ContradictionDetectionState,
            {
                "input_text": "test input",
                "queries": ["test sentence"],
                "regex_retrieved_data": None,
                "ner_retrieved_data": None,
                "semantic_retrieved_data": None,
                "hybrid_retrieved_data": None,
                "retrieved_data_for_queries": None,
                "contradiction_results": None,
                "report": None,
            },
        )

        updated_state = detect_contradictions(state)

        assert "contradiction_results" in updated_state
        # Should handle None gracefully and return empty results


class TestConcurrencyIntegration:
    """Integration tests for concurrent processing."""

    @patch("climatefact.workflows.contradiction_detection.nodes.detect_contradictions.get_nli_client")
    def test_concurrent_processing_with_multiple_sentences_and_passages(self, mock_get_client):
        # Create a mock client that returns different labels based on input
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        def mock_invoke(input_text):
            if "contradiction" in input_text.lower():
                return Mock(text="contradiction")
            elif "entailment" in input_text.lower():
                return Mock(text="entailment")
            else:
                return Mock(text="neutral")

        mock_client.invoke = mock_invoke

        # Prepare complex state with multiple sentences and passages
        state = cast(
            ContradictionDetectionState,
            {
                "input_text": "test input",
                "queries": [
                    "This is a contradiction statement",
                    "This is an entailment statement",
                    "This is a neutral statement",
                ],
                "regex_retrieved_data": None,
                "ner_retrieved_data": None,
                "semantic_retrieved_data": None,
                "hybrid_retrieved_data": None,
                "retrieved_data_for_queries": [
                    [
                        {"text": "passage with contradiction", "source": "source 1"},
                        {"text": "another passage", "source": "source 2"},
                    ],
                    [{"text": "passage with entailment", "source": "source 3"}],
                    [
                        {"text": "normal passage", "source": "source 4"},
                        {"text": "another normal passage", "source": "source 5"},
                    ],
                ],
                "contradiction_results": None,
                "report": None,
            },
        )

        # Execute
        updated_state = detect_contradictions(state)

        # Verify structure
        assert "contradiction_results" in updated_state
        results = updated_state["contradiction_results"]
        assert results is not None
        assert len(results) == 3

        # Each sentence should have been processed
        for result in results:
            assert "sentence" in result
            assert "contradictions" in result
            assert "has_contradictions" in result


@pytest.fixture
def sample_state():
    """Fixture providing a sample ContradictionDetectionState."""
    return cast(
        ContradictionDetectionState,
        {
            "input_text": "test input",
            "queries": ["Climate change is not real", "Global warming is accelerating"],
            "regex_retrieved_data": None,
            "ner_retrieved_data": None,
            "semantic_retrieved_data": None,
            "hybrid_retrieved_data": None,
            "retrieved_data_for_queries": [
                [
                    {"text": "Climate change is a well-documented phenomenon", "source": "IPCC Report"},
                    {"text": "Temperature records show consistent warming", "source": "NASA Data"},
                ],
                [
                    {"text": "Global temperatures are rising rapidly", "source": "Climate Study"},
                    {"text": "Ice sheets are melting at unprecedented rates", "source": "Arctic Research"},
                ],
            ],
            "contradiction_results": None,
            "report": None,
        },
    )


class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    @patch("climatefact.workflows.contradiction_detection.nodes.detect_contradictions.get_nli_client")
    def test_realistic_climate_contradiction_scenario(self, mock_get_client, sample_state):
        """Test with realistic climate data scenario."""
        # Setup mock to simulate contradictions for climate denial statements
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        def realistic_mock_invoke(input_text):
            # Simulate that climate denial statements contradict scientific evidence
            if "not real" in input_text and "well-documented" in input_text:
                return Mock(text="contradiction")
            else:
                return Mock(text="neutral")

        mock_client.invoke = realistic_mock_invoke

        # Execute
        result_state = detect_contradictions(sample_state)

        # Verify
        results = result_state["contradiction_results"]
        assert results is not None
        assert len(results) == 2

        # First sentence should have contradictions detected
        first_result = results[0]
        assert first_result["sentence"] == "Climate change is not real"
        # Note: In real scenario, this would depend on actual NLI model responses


if __name__ == "__main__":
    pytest.main([__file__])
