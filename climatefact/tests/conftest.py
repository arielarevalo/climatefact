"""Shared pytest fixtures for climatefact tests."""

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env() -> None:
    """Load environment variables from .env file for all tests."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


@pytest.fixture
def mock_openai_client() -> Generator[MagicMock, None, None]:
    """Mock OpenAI client for testing without API calls."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Mocked response"))]
    mock_client.chat.completions.create.return_value = mock_response
    yield mock_client


@pytest.fixture
def mock_langgraph_client() -> Generator[MagicMock, None, None]:
    """Mock LangGraph client for testing without API calls."""
    mock_client = MagicMock()
    yield mock_client


@pytest.fixture
def sample_climate_statement() -> str:
    """Sample climate statement for testing."""
    return "Global temperatures have increased by 1.5°C since pre-industrial times."


@pytest.fixture
def sample_contradiction() -> dict[str, Any]:
    """Sample contradiction result for testing."""
    return {
        "has_contradiction": True,
        "contradictory_text": "Global temperatures have not changed significantly.",
        "explanation": "The statement contradicts the scientific consensus on temperature rise.",
        "confidence": 0.85,
    }


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test files."""
    return tmp_path


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock common environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "test-langchain-key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
