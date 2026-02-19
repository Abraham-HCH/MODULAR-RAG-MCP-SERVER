"""
Unit tests for ChunkRefiner.
"""

import pytest
from src.ingestion.transform.chunk_refiner import ChunkRefiner

@pytest.fixture
def sample_chunks():
    """Provide sample chunks for testing."""
    return [
        {"text": "This is   a   test chunk."},
        {"text": "Another   chunk with   extra spaces."}
    ]

def test_rule_based_refinement(sample_chunks):
    """
    Test rule-based refinement in ChunkRefiner.
    """
    refiner = ChunkRefiner(use_llm=False)
    refined_chunks = refiner.process(sample_chunks)

    assert refined_chunks[0]["text"] == "This is a test chunk."
    assert refined_chunks[1]["text"] == "Another chunk with extra spaces."

def test_llm_refinement(sample_chunks):
    """
    Test LLM-based refinement in ChunkRefiner.
    """
    refiner = ChunkRefiner(use_llm=True)
    refined_chunks = refiner.process(sample_chunks)

    assert refined_chunks[0]["text"] == "This is a test chunk. [LLM refined]"
    assert refined_chunks[1]["text"] == "Another chunk with extra spaces. [LLM refined]"