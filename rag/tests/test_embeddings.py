import pytest

from rag.embeddings import EmbeddingModel


@pytest.fixture(scope="module")
def embedding_model():
    return EmbeddingModel()


def test_embed_text_returns_list(embedding_model):
    vector = embedding_model.embed_text(
        "Refund information"
    )

    assert isinstance(vector, list)


def test_embed_text_returns_non_empty_vector(embedding_model):
    vector = embedding_model.embed_text(
        "Refund information"
    )

    assert len(vector) > 0


def test_embedding_dimension_is_consistent(embedding_model):
    vector = embedding_model.embed_text(
        "Refund information"
    )

    assert len(vector) == 384


def test_embed_documents_returns_one_vector_per_text(
    embedding_model,
):
    texts = [
        "Refund information",
        "Order information",
        "Customer support information",
    ]

    vectors = embedding_model.embed_documents(texts)

    assert len(vectors) == len(texts)


def test_all_document_vectors_have_same_dimension(
    embedding_model,
):
    texts = [
        "Refund information",
        "Order information",
        "Customer support information",
    ]

    vectors = embedding_model.embed_documents(texts)

    dimensions = {len(vector) for vector in vectors}

    assert dimensions == {384}


def test_empty_document_list_returns_empty_result(
    embedding_model,
):
    result = embedding_model.embed_documents([])

    assert result == []


def test_empty_text_raises_error(embedding_model):
    with pytest.raises(ValueError):
        embedding_model.embed_text("")


def test_whitespace_text_raises_error(embedding_model):
    with pytest.raises(ValueError):
        embedding_model.embed_text("   ")


def test_empty_document_text_raises_error(
    embedding_model,
):
    with pytest.raises(ValueError):
        embedding_model.embed_documents(
            ["Refund information", ""]
        )