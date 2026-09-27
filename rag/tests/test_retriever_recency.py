from datetime import datetime, timedelta

from rag.retriever import Retriever


class FakeStore:
    def search(self, query_embedding, top_k=3, where=None):
        now = datetime.now()

        return {
            "documents": [[
                "Old document",
                "Recent document",
                "Medium document",
            ]],
            "metadatas": [[
                {
                    "source": "old.md",
                    "chunk_id": "old:0",
                    "created_at": (now - timedelta(days=30)).isoformat(),
                },
                {
                    "source": "recent.md",
                    "chunk_id": "recent:0",
                    "created_at": (now - timedelta(days=1)).isoformat(),
                },
                {
                    "source": "medium.md",
                    "chunk_id": "medium:0",
                    "created_at": (now - timedelta(days=10)).isoformat(),
                },
            ]],
        }


def test_retriever_can_rank_results_by_recency():
    retriever = Retriever(FakeStore(), top_k=3)

    result = retriever.retrieve(
        [0.1, 0.2, 0.3],
        use_recency=True,
    )

    chunk_ids = [
        metadata["chunk_id"]
        for metadata in result["metadatas"][0]
    ]

    assert chunk_ids[0] == "recent:0"


def test_recency_ranking_keeps_documents_and_metadata_together():
    retriever = Retriever(FakeStore(), top_k=3)

    result = retriever.retrieve(
        [0.1, 0.2, 0.3],
        use_recency=True,
    )

    assert len(result["documents"][0]) == 3
    assert len(result["metadatas"][0]) == 3

    for document, metadata in zip(
        result["documents"][0],
        result["metadatas"][0],
    ):
        assert metadata["source"].replace(".md", "").capitalize() in document