import os
from types import SimpleNamespace

import pytest

from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeEmbedding(BaseEmbedding):
    def __init__(self, settings=None, **kwargs):
        pass

    def embed(self, texts, trace=None, **kwargs):
        return [[float(len(t)) * 0.1 for _ in range(4)] for t in texts]

    def get_dimension(self):
        return 4


class FakeStore(BaseVectorStore):
    def __init__(self, settings=None, **kwargs):
        self.storage = {}

    def upsert(self, records, trace=None, **kwargs):
        self.validate_records(records)
        for r in records:
            self.storage[r['id']] = r

    def query(self, vector, top_k=10, filters=None, trace=None, **kwargs):
        return []


def make_settings():
    s = SimpleNamespace()
    e = SimpleNamespace()
    v = SimpleNamespace()
    e.provider = 'fake'
    v.provider = 'fake'
    s.embedding = e
    s.vector_store = v
    return s


def test_ingest_file_upsert_and_history(tmp_path, monkeypatch):
    # register fake providers
    EmbeddingFactory.register_provider('fake', FakeEmbedding)
    VectorStoreFactory.register_provider('fake', FakeStore)

    # create a small pdf-like file (plain text) to ingest
    p = tmp_path / "sample.txt"
    p.write_text("This is a test document.\nSecond line.")

    settings = make_settings()
    # Pipeline expects dict for splitter config; pass object for factories
    pipeline = IngestionPipeline({"splitter_type": "default"})
    # override settings attr for factories
    pipeline.settings = settings

    # monkeypatch load_pdf to return Document without requiring PyPDF2
    class Doc:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata

    def fake_load(path):
        return Doc(text=p.read_text(), metadata={"file_path": str(p)})

    monkeypatch.setattr(pipeline, 'load_pdf', fake_load)

    # Ensure ingestion DB exists
    os.makedirs('data/db', exist_ok=True)

    result = pipeline.ingest_file(str(p))

    assert result['status'] == 'success'
    assert result['chunk_count'] > 0

    # Check ingestion_history entry
    import sqlite3

    conn = sqlite3.connect('data/db/ingestion_history.db')
    cur = conn.cursor()
    cur.execute("SELECT file_hash, status, chunk_count FROM ingestion_history WHERE file_path = ?", (str(p),))
    row = cur.fetchone()
    conn.close()

    assert row is not None
    assert row[1] == 'success'
    assert row[2] == result['chunk_count']


def test_ingest_idempotent_runs_twice(tmp_path, monkeypatch):
    EmbeddingFactory.register_provider('fake', FakeEmbedding)
    VectorStoreFactory.register_provider('fake', FakeStore)

    p = tmp_path / "sample2.txt"
    p.write_text("Idempotent content")

    settings = make_settings()
    pipeline = IngestionPipeline({"splitter_type": "default"})
    pipeline.settings = settings

    class Doc:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata

    def fake_load(path):
        return Doc(text=p.read_text(), metadata={"file_path": str(p)})

    monkeypatch.setattr(pipeline, 'load_pdf', fake_load)

    # First ingest
    r1 = pipeline.ingest_file(str(p))
    # Second ingest (should be idempotent in vector store sense)
    r2 = pipeline.ingest_file(str(p))

    assert r1['chunk_count'] == r2['chunk_count']
    assert r1['file_hash'] == r2['file_hash']
