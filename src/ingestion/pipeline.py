"""Ingestion Pipeline Implementation.

This module defines the ingestion pipeline for processing documents.
"""

from typing import Any, List

from src.libs.splitter.splitter_factory import SplitterFactory

class IngestionPipeline:
    """Pipeline for ingesting and processing documents."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.splitter = SplitterFactory.create(settings)

    def process_document(self, document: str) -> List[str]:
        """Process a document and return chunks.

        Args:
            document: The document to process.

        Returns:
            A list of chunks.
        """
        return self.splitter.split_text(document)
