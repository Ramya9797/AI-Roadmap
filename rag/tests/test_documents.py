from pathlib import Path


DOCUMENT_DIR = Path("rag/documents")


EXPECTED_DOCUMENTS = {
    "refund_policy.md",
    "order_policy.md",
    "customer_support.md",
    "churn_strategy.md",
}


def test_document_directory_exists():
    assert DOCUMENT_DIR.exists()
    assert DOCUMENT_DIR.is_dir()


def test_expected_documents_exist():
    actual_documents = {
        file.name
        for file in DOCUMENT_DIR.glob("*.md")
    }

    assert actual_documents == EXPECTED_DOCUMENTS


def test_documents_are_not_empty():
    documents = list(DOCUMENT_DIR.glob("*.md"))

    assert documents

    for document in documents:
        content = document.read_text(
            encoding="utf-8"
        ).strip()

        assert content, f"{document.name} is empty"


def test_documents_have_title():
    documents = list(DOCUMENT_DIR.glob("*.md"))

    for document in documents:
        content = document.read_text(
            encoding="utf-8"
        ).strip()

        assert content.startswith("# "), (
            f"{document.name} does not have a markdown title"
        )


def test_refund_policy_contains_required_sections():
    content = (
        DOCUMENT_DIR / "refund_policy.md"
    ).read_text(encoding="utf-8")

    assert "## Refund Eligibility" in content
    assert "## Refund Review" in content
    assert "## Escalation" in content


def test_order_policy_contains_required_sections():
    content = (
        DOCUMENT_DIR / "order_policy.md"
    ).read_text(encoding="utf-8")

    assert "## Order Status" in content
    assert "## Customer Response" in content
    assert "## Order Not Found" in content


def test_customer_support_contains_required_sections():
    content = (
        DOCUMENT_DIR / "customer_support.md"
    ).read_text(encoding="utf-8")

    assert "## Repeated Support Issues" in content
    assert "## Investigation" in content
    assert "## Human Review" in content


def test_churn_strategy_contains_required_sections():
    content = (
        DOCUMENT_DIR / "churn_strategy.md"
    ).read_text(encoding="utf-8")

    assert "## Churn Prediction" in content
    assert "## High Risk Customers" in content
    assert "## Supporting Factors" in content