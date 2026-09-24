from rag.eval_dataset import load_eval_dataset


def test_eval_dataset_contains_questions():
    dataset = load_eval_dataset()

    assert len(dataset) >= 5

    for item in dataset:
        assert "question" in item
        assert "relevant_chunk_ids" in item

        assert isinstance(item["question"], str)
        assert item["question"].strip()

        assert isinstance(item["relevant_chunk_ids"], list)
        assert len(item["relevant_chunk_ids"]) > 0


def test_eval_dataset_contains_known_questions():
    dataset = load_eval_dataset()

    questions = [
        item["question"]
        for item in dataset
    ]

    assert "What is the refund policy?" in questions
    assert "How can I check my order status?" in questions


def test_eval_dataset_has_expected_chunk_ids():
    dataset = load_eval_dataset()

    for item in dataset:
        for chunk_id in item["relevant_chunk_ids"]:
            assert isinstance(chunk_id, str)
            assert chunk_id.strip()


def test_eval_dataset_chunk_ids_are_unique_per_question():
    dataset = load_eval_dataset()

    for item in dataset:
        chunk_ids = item["relevant_chunk_ids"]

        assert len(chunk_ids) == len(set(chunk_ids))


from rag.ingest import ingest_documents


def test_eval_dataset_chunk_ids_exist_in_vector_store():
    dataset = load_eval_dataset()

    store = ingest_documents()

    stored_data = store.get_all()

    stored_chunk_ids = set(stored_data["ids"])

    for item in dataset:
        for chunk_id in item["relevant_chunk_ids"]:
            assert chunk_id in stored_chunk_ids