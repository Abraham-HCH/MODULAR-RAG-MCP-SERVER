"""Tests for Transform integration in IngestionPipeline."""

from src.ingestion.pipeline import IngestionPipeline

def test_process_document_with_transforms():
    settings = {
        "splitter_type": "default",
        "transforms": ["chunk_refiner", "metadata_enricher"],
    }

    pipeline = IngestionPipeline(settings)
    doc = "This   is   a   test.\n\nSecond paragraph."
    chunks = pipeline.process_document_with_transforms(doc)

    assert isinstance(chunks, list)
    assert all(isinstance(c, dict) for c in chunks)
    assert all("text" in c and "metadata" in c for c in chunks)
    # ChunkRefiner should normalize whitespace
    assert all("  " not in c["text"] for c in chunks)
    # MetadataEnricher should add title when missing
    assert all("title" in c["metadata"] for c in chunks)
