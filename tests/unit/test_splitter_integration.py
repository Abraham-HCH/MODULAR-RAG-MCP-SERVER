"""
Unit tests for Splitter Integration.
"""

import pytest
from src.ingestion.splitter_integration import SplitterIntegration
from src.core.types import Document
from src.core.settings import Settings
from src.core.settings import LLMSettings, EmbeddingSettings, VectorStoreSettings, RetrievalSettings, RerankSettings, EvaluationSettings, ObservabilitySettings, IngestionSettings

def mock_settings():
    """Mock settings for testing."""
    return Settings(
        llm=LLMSettings(provider="mock", model="mock", temperature=0.7, max_tokens=100),
        embedding=EmbeddingSettings(provider="mock", model="mock", dimensions=128),
        vector_store=VectorStoreSettings(provider="mock", persist_directory="mock_dir", collection_name="mock_collection"),
        retrieval=RetrievalSettings(dense_top_k=10, sparse_top_k=10, fusion_top_k=10, rrf_k=5),
        rerank=RerankSettings(enabled=False, provider="mock", model="mock", top_k=5),
        evaluation=EvaluationSettings(enabled=False, provider="mock", metrics=[]),
        observability=ObservabilitySettings(log_level="INFO", trace_enabled=False, trace_file="mock_trace", structured_logging=False),
        ingestion=IngestionSettings(chunk_size=100, chunk_overlap=10, splitter="recursive", batch_size=10)
    )

def test_splitter_integration():
    """
    Test the SplitterIntegration with a sample document.
    """
    # Example document
    example_document = Document(
        id="example_doc_001",
        text="""# Title\n\n---\n\nThis is a paragraph.\n\n---\n\n## Subtitle\n\n---\n\nAnother paragraph.""",
        metadata={"source_path": "example.pdf"}
    )

    # Initialize the integration with mock settings
    splitter_integration = SplitterIntegration(splitter_type="recursive", settings=mock_settings())

    # Adjust chunk size and overlap for testing
    splitter_integration.splitter.chunk_size = 20
    splitter_integration.splitter.chunk_overlap = 5

    # Process the document
    chunks = splitter_integration.process_document(example_document)

    # Debugging output
    print("Generated chunks:", chunks)

    # Assertions
    assert len(chunks) > 1, "Chunks should be generated."
    
    # Combine all chunk texts and normalize spaces for validation
    normalized_text = " ".join(chunk.text.strip() for chunk in chunks).replace("\n", " ").replace("---", "").strip()
    assert "# Title" in normalized_text, "Normalized text should contain the title."
    assert "This is a paragraph." in normalized_text, "Normalized text should contain the first paragraph."

def test_list_available_splitters():
    """
    Test listing available splitter providers.
    """
    splitter_integration = SplitterIntegration(splitter_type="recursive", settings=mock_settings())

    # Get available splitters
    available_splitters = splitter_integration.list_available_splitters()

    # Assertions
    assert "recursive" in available_splitters, "Recursive splitter should be available."
    assert len(available_splitters) > 0, "There should be at least one splitter available."