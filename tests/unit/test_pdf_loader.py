"""Unit tests for PDF loading in IngestionPipeline."""

import pytest
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline, Document

@pytest.fixture
def sample_pdf(tmp_path):
    """Create a sample PDF file for testing."""
    from PyPDF2 import PdfWriter

    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # Ensure only one blank page is added
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path

def test_load_pdf(sample_pdf):
    """Test loading a PDF file."""
    pipeline = IngestionPipeline(settings={})
    document = pipeline.load_pdf(str(sample_pdf))

    assert isinstance(document, Document)
    assert "file_path" in document.metadata
    assert document.metadata["page_count"] == 1
    assert document.text.strip() == "", "Text should be empty for a blank page."