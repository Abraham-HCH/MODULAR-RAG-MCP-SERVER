"""Unit tests for Ingestion Pipeline with Splitter integration."""

from unittest.mock import MagicMock
from typing import Any, List

import pytest

from src.ingestion.pipeline import IngestionPipeline
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.splitter.base_splitter import BaseSplitter

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
    settings.ingestion = MagicMock()
    settings.ingestion.splitter = "fake"

    pipeline = IngestionPipeline(settings)
    document = "This is a test document."
    chunks = pipeline.process_document(document)

    assert chunks == ["This", "is", "a", "test", "document."]
