"""
Unit tests for MetadataEnricher.
"""

import pytest
from src.ingestion.transform.metadata_enricher import MetadataEnricher

@pytest.fixture
def sample_chunks():
    """Provide sample chunks for testing."""
    return [
        {"text": "This is a test chunk. It contains some text.", "metadata": {}},
        {"text": "Another chunk with more content.", "metadata": {"title": "Existing Title"}}
    ]

def test_rule_based_enrichment(sample_chunks):
    """
    Test rule-based metadata enrichment in MetadataEnricher.
    """
    enricher = MetadataEnricher(use_llm=False)
    enriched_chunks = enricher.process(sample_chunks)

    assert enriched_chunks[0]["metadata"]["title"] == "This is a test chunk"
    assert enriched_chunks[1]["metadata"]["title"] == "Existing Title"

def test_llm_enrichment(sample_chunks):
    """
    Test LLM-based metadata enrichment in MetadataEnricher.
    """
    enricher = MetadataEnricher(use_llm=True)
    enriched_chunks = enricher.process(sample_chunks)

    assert enriched_chunks[0]["metadata"]["summary"] == "This is a test chunk. It contains some text.... [LLM summary]"
    assert enriched_chunks[1]["metadata"]["summary"] == "Another chunk with more content.... [LLM summary]"