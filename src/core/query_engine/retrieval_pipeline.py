"""
Retrieval Pipeline Implementation.

This module implements the hybrid search pipeline, combining dense and sparse retrieval
strategies with result fusion and optional reranking.
"""

from typing import List, Dict, Any

class RetrievalPipeline:
    """Hybrid Retrieval Pipeline."""

    def __init__(self, dense_retriever: Any, sparse_retriever: Any, fusion_strategy: Any, reranker: Any = None):
        """
        Initialize the retrieval pipeline.

        Args:
            dense_retriever: Component for dense retrieval (e.g., embedding-based).
            sparse_retriever: Component for sparse retrieval (e.g., BM25).
            fusion_strategy: Strategy for fusing dense and sparse results.
            reranker: Optional reranker for refining the top results.
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_strategy = fusion_strategy
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval for the given query.

        Args:
            query: The input query string.
            top_k: The number of top results to return.

        Returns:
            A list of retrieved documents with metadata.
        """
        # Perform dense retrieval
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k)

        # Perform sparse retrieval
        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k)

        # Fuse results
        fused_results = self.fusion_strategy.fuse(dense_results, sparse_results)

        # Optionally rerank results
        if self.reranker:
            fused_results = self.reranker.rerank(query, fused_results)

        return fused_results