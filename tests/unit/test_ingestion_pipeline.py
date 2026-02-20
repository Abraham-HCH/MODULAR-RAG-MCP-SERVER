"""Unit tests for IngestionPipeline."""

import pytest
from src.ingestion.pipeline import IngestionPipeline
from src.libs.splitter.splitter_factory import SplitterType
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_settings():
    return {
        "splitter_type": SplitterType.DEFAULT,
        "chunk_size": 100
    }

@pytest.fixture
def sample_document():
    return "This is a sample document. It contains multiple sentences for testing purposes."

def test_ingestion_pipeline_with_default_splitter(mock_settings, sample_document):
    pipeline = IngestionPipeline(mock_settings)
    chunks = pipeline.process_document(sample_document)

    assert len(chunks) > 0, "Chunks should be generated."
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings."
    assert sum(len(chunk) for chunk in chunks) == len(sample_document), "Chunk lengths should match document length."

@patch("src.ingestion.pipeline.IngestionPipeline._calculate_file_hash")
@patch("src.ingestion.pipeline.IngestionPipeline._check_ingestion_history")
def test_deduplication_skips_processed_files(mock_check_history, mock_calculate_hash, mock_settings):
    """Test that the pipeline skips files already processed."""
    pipeline = IngestionPipeline(mock_settings)
    mock_calculate_hash.return_value = "dummyhash"
    mock_check_history.return_value = True

    result = pipeline.load_pdf("dummy_path.pdf")

    assert result is None
    mock_calculate_hash.assert_called_once_with("dummy_path.pdf")
    mock_check_history.assert_called_once_with("dummyhash")

@patch("src.ingestion.pipeline.IngestionPipeline._calculate_file_hash")
@patch("src.ingestion.pipeline.IngestionPipeline._check_ingestion_history")
def test_deduplication_processes_new_files(mock_check_history, mock_calculate_hash, mock_settings):
    """Test that the pipeline processes new files."""
    pipeline = IngestionPipeline(mock_settings)
    mock_calculate_hash.return_value = "newhash"
    mock_check_history.return_value = False

    with patch("src.ingestion.pipeline.PdfReader") as mock_pdf_reader:
        mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: "Page 1 text")]

        result = pipeline.load_pdf("new_path.pdf")

        assert result is not None
        assert result.metadata["file_path"] == "new_path.pdf"
        assert result.metadata["page_count"] == 1
        mock_calculate_hash.assert_called_once_with("new_path.pdf")
        mock_check_history.assert_called_once_with("newhash")