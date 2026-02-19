"""
MetadataEnricher for enhancing chunk metadata.
"""
from typing import List, Dict
from src.ingestion.transform.base_transform import BaseTransform

class MetadataEnricher(BaseTransform):
    """
    Enhances metadata for chunks using rule-based and optional LLM-based methods.
    """

    def __init__(self, use_llm: bool = False):
        """
        Initialize the MetadataEnricher.

        Args:
            use_llm: Whether to use an LLM for metadata enhancement.
        """
        self.use_llm = use_llm

    def process(self, chunks: List[Dict]) -> List[Dict]:
        """
        Enhance metadata for chunks.

        Args:
            chunks: A list of chunk dictionaries to process.

        Returns:
            A list of chunks with enhanced metadata.
        """
        enriched_chunks = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            # Rule-based metadata enhancement
            metadata = self._rule_based_enrichment(metadata, chunk.get("text", ""))

            # Optional LLM-based metadata enhancement
            if self.use_llm:
                metadata = self._llm_enrichment(metadata, chunk.get("text", ""))

            chunk["metadata"] = metadata
            enriched_chunks.append(chunk)

        return enriched_chunks

    def _rule_based_enrichment(self, metadata: Dict, text: str) -> Dict:
        """
        Apply rule-based metadata enrichment.

        Args:
            metadata: The existing metadata.
            text: The chunk text.

        Returns:
            The enriched metadata.
        """
        # Example: Add a simple title based on the first sentence
        if "title" not in metadata:
            metadata["title"] = text.split(".")[0] if text else "Untitled"
        return metadata

    def _llm_enrichment(self, metadata: Dict, text: str) -> Dict:
        """
        Apply LLM-based metadata enrichment.

        Args:
            metadata: The existing metadata.
            text: The chunk text.

        Returns:
            The enriched metadata.
        """
        # Placeholder for LLM call
        metadata["summary"] = text[:50] + "... [LLM summary]"
        return metadata