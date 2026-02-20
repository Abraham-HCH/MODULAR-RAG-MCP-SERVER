from src.ingestion.pipeline import IngestionPipeline


def test_prepare_records_for_upsert_stable_ids():
    settings = {}
    pipeline = IngestionPipeline(settings)

    chunks = [
        {"text": "This is a chunk.", "metadata": {"chunk_index": 0}},
        {"text": "Another chunk.", "metadata": {"chunk_index": 1}},
    ]

    records1 = pipeline.prepare_records_for_upsert(chunks, embeddings=None, id_prefix="doc1")
    records2 = pipeline.prepare_records_for_upsert(chunks, embeddings=None, id_prefix="doc1")

    assert len(records1) == 2
    assert len(records2) == 2

    # IDs should be stable across calls
    ids1 = [r["id"] for r in records1]
    ids2 = [r["id"] for r in records2]
    assert ids1 == ids2

    # Different prefix changes ids
    records3 = pipeline.prepare_records_for_upsert(chunks, embeddings=None, id_prefix="doc2")
    ids3 = [r["id"] for r in records3]
    assert ids3 != ids1


def test_prepare_records_for_upsert_with_embeddings():
    settings = {}
    pipeline = IngestionPipeline(settings)

    chunks = [{"text": "Embed me.", "metadata": {"chunk_index": 0}}]
    embeddings = [[0.1, 0.2, 0.3]]

    records = pipeline.prepare_records_for_upsert(chunks, embeddings=embeddings)

    assert records[0]["vector"] == embeddings[0]
    assert "id" in records[0]
    assert records[0]["metadata"]["text"] == "Embed me."
