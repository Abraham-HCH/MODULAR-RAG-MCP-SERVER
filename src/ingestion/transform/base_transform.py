"""
Base class for Transform operations in the ingestion pipeline.
"""
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseTransform(ABC):
    """
    Abstract base class for all Transform operations.
    """

    @abstractmethod
    def process(self, chunks: List[Dict]) -> List[Dict]:
        """
        Process a list of chunks and return transformed chunks.

        Args:
            chunks: A list of chunk dictionaries to process.

        Returns:
            A list of transformed chunk dictionaries.
        """
        pass