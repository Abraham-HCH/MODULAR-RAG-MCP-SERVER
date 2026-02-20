"""Unit tests for deduplication mechanism in IngestionPipeline."""

import pytest
from unittest.mock import patch, mock_open
from src.ingestion.pipeline import IngestionPipeline

@pytest.fixture
def pipeline():
    """Create an instance of IngestionPipeline for testing."""
    settings = {"splitter_type": "default"}
    return IngestionPipeline(settings)

@patch("src.ingestion.pipeline.hashlib.sha256")
@patch("src.ingestion.pipeline.open", new_callable=mock_open, read_data=b"test content")
def test_calculate_file_hash(mock_file, mock_hash, pipeline):
    """Test file hash calculation."""
    mock_hash.return_value.hexdigest.return_value = "fakehash"
    file_hash = pipeline._calculate_file_hash("dummy_path")
    assert file_hash == "fakehash"
    mock_file.assert_called_once_with("dummy_path", "rb")

@patch("src.ingestion.pipeline.sqlite3.connect")
def test_check_ingestion_history(mock_connect, pipeline):
    """Test ingestion history check."""
    mock_cursor = mock_connect.return_value.cursor.return_value
    mock_cursor.fetchone.return_value = ("success",)

    file_hash = "fakehash"
    result = pipeline._check_ingestion_history(file_hash)

    mock_connect.assert_called_once_with("data/db/ingestion_history.db")
    mock_cursor.execute.assert_called_once_with(
        """
        SELECT status FROM ingestion_history
        WHERE file_hash = ? AND status = 'success'
        """,
        (file_hash,)
    )
    assert result is True

@patch("src.ingestion.pipeline.IngestionPipeline._check_ingestion_history")
@patch("src.ingestion.pipeline.IngestionPipeline._calculate_file_hash")
def test_load_pdf_deduplication(mock_calculate_hash, mock_check_history, pipeline):
    """Test load_pdf skips processing for duplicate files."""
    mock_calculate_hash.return_value = "fakehash"
    mock_check_history.return_value = True

    result = pipeline.load_pdf("dummy_path")

    mock_calculate_hash.assert_called_once_with("dummy_path")
    mock_check_history.assert_called_once_with("fakehash")
    assert result is None