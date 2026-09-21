from pathlib import Path

from rag.loader import load_documents


def test_load_documents_returns_four_documents():
    documents = load_documents()

    assert len(documents) == 4


def test_loaded_documents_have_source():
    documents = load_documents()

    for document in documents:
        assert "source" in document
        assert document["source"]


def test_loaded_documents_have_text():
    documents = load_documents()

    for document in documents:
        assert "text" in document
        assert document["text"]


def test_loaded_documents_have_expected_sources():
    documents = load_documents()

    sources = {
        document["source"]
        for document in documents
    }

    expected_sources = {
        "refund_policy.md",
        "order_policy.md",
        "customer_support.md",
        "churn_strategy.md",
    }

    assert sources == expected_sources


def test_loaded_documents_are_sorted():
    documents = load_documents()

    sources = [
        document["source"]
        for document in documents
    ]

    assert sources == sorted(sources)


def test_loaded_content_is_string():
    documents = load_documents()

    for document in documents:
        assert isinstance(document["text"], str)


def test_loader_skips_empty_files(tmp_path, monkeypatch):
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("", encoding="utf-8")

    valid_file = tmp_path / "valid.md"
    valid_file.write_text(
        "# Valid Document\n\nSome content.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "rag.loader.DOCUMENT_DIR",
        tmp_path,
    )

    documents = load_documents()

    assert len(documents) == 1
    assert documents[0]["source"] == "valid.md"


def test_loader_ignores_non_markdown_files(tmp_path, monkeypatch):
    markdown_file = tmp_path / "valid.md"
    markdown_file.write_text(
        "# Valid\n\nContent.",
        encoding="utf-8",
    )

    text_file = tmp_path / "ignored.txt"
    text_file.write_text(
        "This should not be loaded.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "rag.loader.DOCUMENT_DIR",
        tmp_path,
    )

    documents = load_documents()

    assert len(documents) == 1
    assert documents[0]["source"] == "valid.md"