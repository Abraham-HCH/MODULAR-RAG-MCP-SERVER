"""Recursive Splitter implementation using LangChain.

This module provides a recursive character-based text splitting strategy
that respects document structure (headers, code blocks) and splits text
hierarchically to maintain semantic coherence.
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore[misc, assignment]

from src.libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """Recursive character-based text splitter.
    
    This splitter uses LangChain's RecursiveCharacterTextSplitter to split text
    by trying different separators in order (paragraphs, sentences, words) while
    respecting Markdown structure elements like headers and code blocks.
    
    Design Principles Applied:
    - Pluggable: Implements BaseSplitter interface for factory instantiation.
    - Config-Driven: Reads chunk_size and chunk_overlap from settings.
    - Fail-Fast: Raises ImportError if langchain-text-splitters is not installed.
    - Graceful Degradation: Validates inputs and provides clear error messages.
    
    Attributes:
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        separators: List of separators to try in order (defaults to Markdown-aware).
        
    Raises:
        ImportError: If langchain-text-splitters package is not installed.
    """
    
    DEFAULT_SEPARATORS = [
        "\n\n",  # Double newline (paragraphs)
        "\n---\n",  # Horizontal rule (Markdown header separation)
        "\n",    # Single newline
        ". ",    # Sentence endings
        "! ",
        "? ",
        "; ",
        ", ",
        " ",     # Spaces
        "",      # Characters
    ]
    
    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RecursiveSplitter.
        
        Args:
            settings: Application settings containing ingestion configuration.
            chunk_size: Optional override for chunk size (defaults to settings.ingestion.chunk_size).
            chunk_overlap: Optional override for overlap (defaults to settings.ingestion.chunk_overlap).
            separators: Optional list of separator strings (defaults to Markdown-aware separators).
            **kwargs: Additional parameters passed to LangChain splitter.
        
        Raises:
            ImportError: If langchain-text-splitters is not installed.
            ValueError: If chunk_size or chunk_overlap are invalid.
        """
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is not installed. "
                "Install it with: pip install langchain-text-splitters"
            )
        
        self.settings = settings
        
        # Extract configuration from settings with overrides
        try:
            ingestion_config = settings.ingestion
            self.chunk_size = chunk_size if chunk_size is not None else ingestion_config.chunk_size
            self.chunk_overlap = chunk_overlap if chunk_overlap is not None else ingestion_config.chunk_overlap
        except AttributeError as e:
            raise ValueError(
                "Missing ingestion configuration in settings. "
                "Expected settings.ingestion.chunk_size and settings.ingestion.chunk_overlap"
            ) from e
        
        # Validate configuration
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {self.chunk_size}")
        
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got: {self.chunk_overlap}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        # Adjust separators to prioritize Markdown structure and avoid splitting within paragraphs
        self.separators = separators if separators is not None else ["\n---\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

        # Ensure chunk size and overlap are reasonable for Markdown
        self.chunk_size = max(self.chunk_size, 50)  # Minimum chunk size
        self.chunk_overlap = min(self.chunk_overlap, self.chunk_size // 3)  # Overlap should not exceed one-third of the chunk size

        # Merge logic is provided via `merge_chunks` instance method (defined below)
        
        # Initialize the underlying langchain splitter instance
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def merge_chunks(self, chunks: List[str], finalize_all: bool = True) -> List[str]:
        """Merge small chunks into larger ones according to `chunk_size`.

        Exposed as a public helper for tests and advanced usage.
        """
        merged = []
        buffer = ""

        def _words(text: str) -> List[str]:
            return [w for w in text.split() if w]

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            if not buffer:
                buffer = chunk
                continue

            # If buffer + chunk fits, merge with overlap removal at word boundary
            if len(buffer) + len(chunk) <= self._chunk_size:
                buf_words = _words(buffer)
                chunk_words = _words(chunk)

                # find maximal overlap in words
                max_k = min(len(buf_words), len(chunk_words))
                overlap = 0
                for k in range(max_k, 0, -1):
                    if buf_words[-k:] == chunk_words[:k]:
                        overlap = k
                        break

                if overlap:
                    combined = buf_words + chunk_words[overlap:]
                else:
                    combined = buf_words + chunk_words

                buffer = " ".join(combined)
            else:
                # Before emitting buffer, remove any overlapping prefix from the
                # next chunk to avoid duplicated word sequences across boundaries.
                buf_words = _words(buffer)
                chunk_words = _words(chunk)
                max_k = min(len(buf_words), len(chunk_words))
                overlap = 0
                for k in range(max_k, 0, -1):
                    if buf_words[-k:] == chunk_words[:k]:
                        overlap = k
                        break

                if overlap:
                    # remove overlapping prefix from next chunk
                    chunk_words = chunk_words[overlap:]
                    chunk = " ".join(chunk_words)

                merged.append(buffer.strip())
                buffer = chunk

        if buffer:
            merged.append(buffer.strip())

        # Final pass: if all parts are small enough that their concatenation
        # still fits within chunk_size, merge them into a single chunk.
        try:
            total_len = sum(len(chunk) for chunk in merged)
            if finalize_all:
                # Old behavior: merge everything into one chunk
                if len(merged) > 1:
                    merged = [" ".join(merged)]
            else:
                # Conservative behavior for runtime splitting: only merge if total fits
                if len(merged) > 1 and total_len <= self._chunk_size:
                    merged = [" ".join(merged)]
        except Exception:
            pass

        return merged

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self._chunk_size = value
        try:
            # Recreate underlying splitter to ensure new size takes effect
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                separators=self.separators,
            )
        except Exception:
            pass

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    @chunk_overlap.setter
    def chunk_overlap(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        self._chunk_overlap = value
        try:
            # Recreate underlying splitter to ensure new overlap takes effect
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                separators=self.separators,
            )
        except Exception:
            pass
        
    def validate_text(self, text: Any) -> None:
        """Validate input text."""
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace.")

    def validate_chunks(self, chunks: List[str]) -> None:
        """Validate output chunks."""
        if not all(isinstance(chunk, str) for chunk in chunks):
            raise ValueError("All chunks must be strings.")
        if not all(chunk.strip() for chunk in chunks):
            raise ValueError("Chunks cannot be empty or whitespace.")

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Split text into chunks recursively.
        
        This method splits text by trying different separators hierarchically,
        preserving document structure like Markdown headers and code blocks.
        
        Args:
            text: Input text to split. Must be a non-empty string.
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Additional parameters (currently unused, reserved for future extensions).
        
        Returns:
            A list of text chunks. Each chunk respects the configured chunk_size
            and chunk_overlap. Order preserves the original text sequence.
        
        Raises:
            ValueError: If input text is invalid (empty, wrong type).
            RuntimeError: If splitting fails unexpectedly.
        
        Example:
            >>> splitter = RecursiveSplitter(settings)
            >>> chunks = splitter.split_text("# Header\\n\\nParagraph 1.\\n\\nParagraph 2.")
            >>> len(chunks)
            1  # If text fits in chunk_size
        """
        # Validate input
        self.validate_text(text)
        
        try:
            # Perform splitting
            chunks = self._splitter.split_text(text)

            # Handle edge case: LangChain may return empty list for very short text
            if not chunks:
                chunks = [text]

            # Merge small/overlapping chunks to improve coherence
            try:
                # Use conservative merge during runtime splitting to avoid
                # collapsing many chunks into a single huge chunk.
                chunks = self.merge_chunks(chunks, finalize_all=False)
            except Exception:
                # If merge fails, fall back to raw chunks
                pass

            # Validate output
            self.validate_chunks(chunks)

            return chunks
            
        except Exception as e:
            # Catch any LangChain errors and provide context
            raise RuntimeError(
                f"RecursiveSplitter failed to split text: {e}. "
                f"Text length: {len(text)}, chunk_size: {self.chunk_size}, "
                f"chunk_overlap: {self.chunk_overlap}"
            ) from e

    # NOTE: Duplicate/simple split_text removed. Use the validated split_text above.
