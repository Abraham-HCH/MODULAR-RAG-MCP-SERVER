"""Ingestion Pipeline Implementation.

This module defines the ingestion pipeline for processing documents.
"""

from typing import Any, List
from pathlib import Path
import hashlib
import sqlite3
import os

from src.libs.splitter.splitter_factory import SplitterFactory, SplitterType
from src.observability.logger import get_logger
from importlib import import_module
from types import SimpleNamespace

logger = get_logger(__name__)

# Expose PdfReader at module level so tests can patch it. Import is optional.
try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None

class Document:
    """Represents a document with text and metadata."""
    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata

class IngestionPipeline:
    """Pipeline for ingesting and processing documents."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        splitter_type = settings.get("splitter_type", SplitterType.DEFAULT)
        logger.info(f"Initializing Splitter of type: {splitter_type}")
        self.splitter = SplitterFactory.create(splitter_type, settings)

    def process_document(self, document: str) -> List[str]:
        """Process a document and return chunks.

        Args:
            document: The document to process.

        Returns:
            A list of chunks.
        """
        logger.info("Processing document...")
        chunks = self.splitter.split_text(document)
        logger.info(f"Generated {len(chunks)} chunks.")
        return chunks

    def process_document_with_transforms(self, document: str) -> List[dict]:
        """Process a document, run splitter, then apply configured transforms.

        Returns a list of chunk dicts: {"text": str, "metadata": dict}.
        """
        chunks = self.process_document(document)

        # Build initial chunk dicts
        chunk_dicts = []
        for i, c in enumerate(chunks):
            chunk_dicts.append({"text": c, "metadata": {"chunk_index": i}})

        # Resolve transforms configuration (supports dict or object settings)
        transforms = []
        if isinstance(self.settings, dict):
            transforms = self.settings.get("transforms", []) or []
            transform_options = self.settings.get("transform_options", {})
        else:
            transforms = getattr(self.settings, "transforms", []) or []
            transform_options = getattr(self.settings, "transform_options", {})

        # Registry mapping simple names to module/class
        registry = {
            "chunk_refiner": ("src.ingestion.transform.chunk_refiner", "ChunkRefiner"),
            "metadata_enricher": ("src.ingestion.transform.metadata_enricher", "MetadataEnricher"),
        }

        for name in transforms:
            key = name.lower()
            if key not in registry:
                logger.warning(f"Unknown transform '{name}', skipping.")
                continue
            module_path, class_name = registry[key]
            try:
                mod = import_module(module_path)
                cls = getattr(mod, class_name)
                opts = transform_options.get(key, {}) if isinstance(transform_options, dict) else {}
                transformer = cls(**opts) if opts else cls()
                chunk_dicts = transformer.process(chunk_dicts)
            except Exception as e:
                logger.exception(f"Failed to apply transform '{name}': {e}")

        return chunk_dicts

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate the SHA256 hash of a file."""
        logger.info(f"Calculating hash for file: {file_path}")
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _as_settings_object(self):
        """Return a settings-like object for factories (supports dict or object)."""
        if not isinstance(self.settings, dict):
            return self.settings

        # Build minimal object with nested attrs if provided in dict
        obj = SimpleNamespace()
        emb = SimpleNamespace()
        vs = SimpleNamespace()

        embedding_conf = self.settings.get("embedding", {})
        vector_conf = self.settings.get("vector_store", {})

        emb.provider = embedding_conf.get("provider")
        emb.model = embedding_conf.get("model")

        vs.provider = vector_conf.get("provider")
        vs.collection_name = vector_conf.get("collection_name")
        vs.persist_directory = vector_conf.get("persist_directory")

        obj.embedding = emb
        obj.vector_store = vs
        return obj

    def ingest_file(self, file_path: str) -> dict:
        """Full ingestion: load file, embed chunks, upsert to vector store, record history.

        Returns a summary dict with keys: file_hash, chunk_count, status.
        """
        doc = self.load_pdf(file_path)
        if doc is None:
            return {"skipped": True}

        # Split and apply transforms
        chunk_dicts = self.process_document_with_transforms(doc.text)

        # Compute file hash used as id prefix
        try:
            file_hash = self._calculate_file_hash(file_path)
        except Exception:
            file_hash = None

        # Create embedding provider and compute embeddings
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        settings_obj = self._as_settings_object()

        embedding_provider = EmbeddingFactory.create(settings_obj)
        texts = [c.get("text", "") for c in chunk_dicts]
        embeddings = embedding_provider.embed(texts)

        # Prepare records and upsert
        records = self.prepare_records_for_upsert(chunk_dicts, embeddings=embeddings, id_prefix=file_hash)

        vector_store = VectorStoreFactory.create(settings_obj)
        vector_store.upsert(records, id_prefix=file_hash)

        # Write ingestion history
        db_path = "data/db/ingestion_history.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else None
            cur.execute(
                """
INSERT OR REPLACE INTO ingestion_history (file_hash, file_path, file_size, status, chunk_count)
VALUES (?, ?, ?, ?, ?)
""",
                (file_hash, file_path, file_size, "success", len(records)),
            )
            conn.commit()
        finally:
            conn.close()

        return {"file_hash": file_hash, "chunk_count": len(records), "status": "success"}

    def _check_ingestion_history(self, file_hash: str) -> bool:
        """Check if the file hash exists in the ingestion history."""
        db_path = "data/db/ingestion_history.db"
        logger.info("Checking ingestion history...")
        # Ensure directory exists to avoid OperationalError on open
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
        SELECT status FROM ingestion_history
        WHERE file_hash = ? AND status = 'success'
        """,
                    (file_hash,)
                )
                result = cursor.fetchone()
                return result is not None
            except sqlite3.OperationalError:
                # Table may not exist in test environments; treat as not-processed
                logger.info("ingestion_history table missing; treating as not processed")
                return False
        finally:
            conn.close()

    def load_pdf(self, file_path: str) -> Document:
        """Load a PDF file and return a Document object.

        Args:
            file_path: The path to the PDF file.

        Returns:
            A Document object containing the text and metadata.
        """
        file_hash = self._calculate_file_hash(file_path)
        if self._check_ingestion_history(file_hash):
            logger.info("File already processed. Skipping ingestion.")
            return None

        logger.info(f"Loading PDF file: {file_path}")
        # Use module-level PdfReader (can be patched in tests). Ensure available.
        if PdfReader is None:
            raise RuntimeError("PyPDF2 is required to load PDFs")

        pdf_reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() for page in pdf_reader.pages)
        metadata = {
            "file_path": file_path,
            "title": Path(file_path).stem,
            "page_count": len(pdf_reader.pages),
        }
        logger.info(f"Loaded PDF with {metadata['page_count']} pages.")
        return Document(text=text, metadata=metadata)

    def prepare_records_for_upsert(
        self,
        chunk_dicts: List[dict],
        embeddings: List[List[float]] | None = None,
        id_prefix: str | None = None,
    ) -> List[dict]:
        """Prepare vector-store records from chunk dicts.

        Generates stable chunk-level IDs using SHA256 over the chunk text.

        Args:
            chunk_dicts: List of chunk dicts with keys `text` and `metadata`.
            embeddings: Optional list of embedding vectors aligned with chunks.
            id_prefix: Optional prefix to namespace IDs (e.g., document hash).

        Returns:
            List of records suitable for VectorStore.upsert(), each with:
                - 'id': stable str id (sha256)
                - 'vector': embedding vector if provided
                - 'metadata': metadata dict including original chunk metadata
        """
        records = []
        for idx, chunk in enumerate(chunk_dicts):
            text = (chunk.get("text") or "").strip()
            hasher = hashlib.sha256()
            # Use canonicalized text as the input for stable ids
            hasher.update(text.encode("utf-8"))
            hex_digest = hasher.hexdigest()

            record_id = f"{id_prefix + '_' if id_prefix else ''}{hex_digest}"

            record: dict = {"id": record_id, "metadata": dict(chunk.get("metadata", {}))}

            if embeddings:
                try:
                    record["vector"] = embeddings[idx]
                except Exception:
                    # If embeddings length mismatches, leave vector absent; caller must validate
                    record["vector"] = None

            # Include chunk text in metadata to help store/search pipelines
            record["metadata"]["text"] = text

            records.append(record)

        return records
