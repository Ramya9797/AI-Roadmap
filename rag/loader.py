from pathlib import Path


DOCUMENT_DIR = Path(__file__).parent / "documents"


def load_documents():
    documents = []

    for path in sorted(DOCUMENT_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append(
            {
                "source": path.name,
                "text": text,
            }
        )

    return documents