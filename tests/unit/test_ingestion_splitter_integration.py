"""Unit tests for Ingestion Pipeline with Splitter integration."""

from unittest.mock import MagicMock
from typing import Any, List

import pytest

from src.ingestion.pipeline import IngestionPipeline
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.splitter_factory import SplitterType

class FakeSplitter(BaseSplitter):
    def __init__(self, settings: Any = None, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs

    def split_text(self, text: str, **kwargs: Any) -> List[str]:
        return text.split(" ")

def test_ingestion_pipeline_with_splitter():
    """Test that the ingestion pipeline integrates with the splitter correctly."""
    SplitterFactory.register_provider("fake", FakeSplitter)

    settings = MagicMock()
    settings.get.return_value = SplitterType.FAKE  # Return proper SplitterType

    pipeline = IngestionPipeline(settings)
    document = "This is a test document."
    chunks = pipeline.process_document(document)

    assert chunks == ["This", "is", "a", "test", "document."]

def test_ingestion_pipeline_chunk_size(mock_settings):
    """
    Test that changing the chunk size in settings affects the output of the pipeline.
    """
    mock_settings.get.return_value = SplitterType.FAKE  # Return proper SplitterType
    mock_settings.ingestion.chunk_size = 10
    pipeline = IngestionPipeline(settings=mock_settings)

    document = "This is a longer test document to validate chunk size changes."
    chunks = pipeline.process_document(document)

    # Assertions
    assert len(chunks) > 1, "Pipeline should produce multiple chunks for smaller chunk size."
    assert all(len(chunk) <= 10 for chunk in chunks), "Each chunk should respect the chunk size."

def test_ingestion_pipeline_with_recursive_splitter():
    """Test that the ingestion pipeline integrates with RecursiveSplitter correctly."""
    SplitterFactory.register_provider("recursive", RecursiveSplitter)

    settings = MagicMock()
    settings.get.return_value = SplitterType.RECURSIVE  # Return proper SplitterType
    settings.ingestion.chunk_size = 100
    settings.ingestion.chunk_overlap = 10

    pipeline = IngestionPipeline(settings)
    document = "# Header\n\nThis is a test document.\n\nAnother paragraph."
    chunks = pipeline.process_document(document)

    assert len(chunks) > 0, "Pipeline should produce chunks."
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings."

def test_ingestion_pipeline_with_invalid_splitter():
    """Test that the ingestion pipeline raises an error for an invalid splitter."""
    settings = MagicMock()
    settings.ingestion = MagicMock()
    settings.ingestion.splitter = "invalid"

    with pytest.raises(ValueError, match="Unsupported Splitter provider"):
        IngestionPipeline(settings)

@pytest.fixture
def mock_settings():
    """Mock settings for testing IngestionPipeline."""
    class MockIngestion:
        chunk_size = 50
        chunk_overlap = 10
        splitter = "fake"

    settings = MagicMock()
    settings.ingestion = MockIngestion()
    return settings

def test_ingestion_pipeline_with_sample_document():
    """Test the ingestion pipeline with a real sample document."""
    from pathlib import Path

    # Load the sample document
    sample_path = Path("tests/fixtures/sample_documents/sample.txt")
    with sample_path.open("r", encoding="utf-8") as file:
        document = file.read()

    # Register and use RecursiveSplitter
    SplitterFactory.register_provider("recursive", RecursiveSplitter)

    settings = MagicMock()
    settings.get.return_value = SplitterType.RECURSIVE  # Return proper SplitterType
    settings.ingestion.chunk_size = 100
    settings.ingestion.chunk_overlap = 10

    pipeline = IngestionPipeline(settings)
    chunks = pipeline.process_document(document)

    # Assertions
    assert len(chunks) > 1, "Pipeline should produce multiple chunks."
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings."
    assert any("RAG" in chunk for chunk in chunks), "Chunks should contain content from the document."
