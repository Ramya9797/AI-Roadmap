from rag.ingest import ingest_documents


def test_ingestion_is_idempotent():
    """
    Running ingestion multiple times should produce
    the same vector-store records without duplicates.
    """

    store1 = ingest_documents()
    ids1 = sorted(store1.get_all()["ids"])

    store2 = ingest_documents()
    ids2 = sorted(store2.get_all()["ids"])

    assert ids1 == ids2
    assert store2.count() == store1.count()