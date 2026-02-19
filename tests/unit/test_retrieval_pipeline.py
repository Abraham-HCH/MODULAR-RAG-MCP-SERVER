"""
Unit tests for RetrievalPipeline.
"""

from unittest.mock import MagicMock
import pytest
from src.core.query_engine.retrieval_pipeline import RetrievalPipeline

def test_retrieval_pipeline_hybrid_search():
    """Test the hybrid search functionality of the retrieval pipeline."""
    # Mock components
    dense_retriever = MagicMock()
    sparse_retriever = MagicMock()
    fusion_strategy = MagicMock()
    reranker = MagicMock()

    # Mock return values
    dense_retriever.retrieve.return_value = [
        {"id": "dense1", "score": 0.9},
        {"id": "dense2", "score": 0.8},
    ]
    sparse_retriever.retrieve.return_value = [
        {"id": "sparse1", "score": 0.85},
        {"id": "sparse2", "score": 0.75},
    ]
    fusion_strategy.fuse.return_value = [
        {"id": "dense1", "score": 0.9},
        {"id": "sparse1", "score": 0.85},
        {"id": "dense2", "score": 0.8},
        {"id": "sparse2", "score": 0.75},
    ]
    reranker.rerank.return_value = [
        {"id": "dense1", "score": 0.95},
        {"id": "sparse1", "score": 0.9},
        {"id": "dense2", "score": 0.85},
        {"id": "sparse2", "score": 0.8},
    ]

    # Initialize pipeline
    pipeline = RetrievalPipeline(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion_strategy=fusion_strategy,
        reranker=reranker,
    )

    # Execute retrieval
    query = "test query"
    results = pipeline.retrieve(query, top_k=4)

    # Assertions
    dense_retriever.retrieve.assert_called_once_with(query, top_k=4)
    sparse_retriever.retrieve.assert_called_once_with(query, top_k=4)
    fusion_strategy.fuse.assert_called_once()
    reranker.rerank.assert_called_once()

    assert len(results) == 4
    assert results[0]["id"] == "dense1"
    assert results[1]["id"] == "sparse1"