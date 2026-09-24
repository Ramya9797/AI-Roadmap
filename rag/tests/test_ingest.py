from rag.ingest import ingest_documents


def test_ingestion_returns_store():
    store = ingest_documents()

    assert store is not None


def test_ingestion_creates_records():
    store = ingest_documents()

    assert store.count() == 9


def test_ingestion_creates_documents():
    store = ingest_documents()

    result = store.get_all()

    assert len(result["documents"]) == 9


def test_ingestion_creates_ids():
    store = ingest_documents()

    result = store.get_all()

    assert len(result["ids"]) == 9


def test_ingestion_creates_metadata():
    store = ingest_documents()

    result = store.get_all()

    assert len(result["metadatas"]) == 9


def test_ingestion_ids_are_unique():
    store = ingest_documents()

    result = store.get_all()

    ids = result["ids"]

    assert len(ids) == len(set(ids))


def test_ingestion_metadata_contains_source():
    store = ingest_documents()

    result = store.get_all()

    for metadata in result["metadatas"]:
        assert "source" in metadata


def test_ingestion_metadata_contains_document_type():
    store = ingest_documents()

    result = store.get_all()

    for metadata in result["metadatas"]:
        assert "document_type" in metadata

def test_ingestion_metadata_contains_chunk_id():
    store = ingest_documents()
    result = store.get_all()

    for metadata in result["metadatas"]:
        assert "chunk_id" in metadata


def test_ingestion_metadata_contains_chunk_index():
    store = ingest_documents()
    result = store.get_all()

    for metadata in result["metadatas"]:
        assert "chunk_index" in metadata


def test_ingestion_metadata_contains_total_chunks():
    store = ingest_documents()
    result = store.get_all()

    for metadata in result["metadatas"]:
        assert "total_chunks" in metadata


def test_ingestion_metadata_is_deterministic():
    store1 = ingest_documents()
    result1 = store1.get_all()

    store2 = ingest_documents()
    result2 = store2.get_all()

    metadata1 = sorted(
        result1["metadatas"],
        key=lambda item: item["chunk_id"],
    )

    metadata2 = sorted(
        result2["metadatas"],
        key=lambda item: item["chunk_id"],
    )

    assert metadata1 == metadata2