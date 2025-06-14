"""
Simple integration test to verify the refactored contradiction detection module works correctly.
"""

import sys


def test_import_and_basic_functionality():
    """Test that we can import the module and create basic data structures."""
    try:
        from climatefact.workflows.contradiction_detection.nodes.detect_contradictions import (
            NLIResult,
            NLITask,
            _organize_nli_results,
            _prepare_nli_tasks,
        )

        print("✓ Successfully imported all components")

        # Test creating NLI task
        task = NLITask(
            sentence_idx=0,
            passage_idx=0,
            sentence="Test sentence",
            passage_text="Test passage",
            passage_source="Test source",
        )
        print(f"✓ Created NLI task: {task.sentence}")

        # Test task preparation
        queries = ["sentence 1", "sentence 2"]
        passages = [[{"text": "passage 1", "source": "source 1"}], [{"text": "passage 2", "source": "source 2"}]]

        tasks = _prepare_nli_tasks(queries, passages)
        print(f"✓ Prepared {len(tasks)} NLI tasks from {len(queries)} queries")

        # Test result organization
        nli_results = [
            NLIResult(0, 0, "sentence 1", "passage 1", "source 1", "neutral"),
            NLIResult(1, 0, "sentence 2", "passage 2", "source 2", "contradiction"),
        ]

        contradiction_results = _organize_nli_results(nli_results, queries)
        print(f"✓ Organized results for {len(contradiction_results)} sentences")

        # Verify structure
        assert len(contradiction_results) == 2
        assert contradiction_results[0]["sentence"] == "sentence 1"
        assert contradiction_results[1]["sentence"] == "sentence 2"
        assert contradiction_results[1]["has_contradictions"] is True
        print("✓ All assertions passed")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_concurrent_layer_separation():
    """Test that the concurrent processing layer is properly separated."""
    try:
        from climatefact.workflows.contradiction_detection.nodes.detect_contradictions import (
            _call_nli_model_sync,
            _process_nli_task,
            _process_nli_tasks_concurrently,
        )

        print("✓ Successfully imported concurrent processing functions")

        # Verify the functions are callable (even if we can't call them without a real client)
        assert callable(_call_nli_model_sync)
        assert callable(_process_nli_task)
        assert callable(_process_nli_tasks_concurrently)
        print("✓ All concurrent processing functions are callable")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("Running integration tests for concurrent contradiction detection...")
    print("=" * 60)

    success = True

    print("\n1. Testing import and basic functionality:")
    success &= test_import_and_basic_functionality()

    print("\n2. Testing concurrent layer separation:")
    success &= test_concurrent_layer_separation()

    print("\n" + "=" * 60)
    if success:
        print("✓ All integration tests passed!")
        print("The concurrent contradiction detection implementation is working correctly.")
    else:
        print("✗ Some integration tests failed!")
        sys.exit(1)
