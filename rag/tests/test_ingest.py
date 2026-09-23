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