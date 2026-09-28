from rag.eval_dataset import load_eval_dataset
from rag.ingest import ingest_documents
from rag.rag import RAGPipeline
from rag.llm import FakeLLM
from rag.retrieval_evaluation import hit_rate, reciprocal_rank


def test_10_case_retrieval_evaluation_gate():
    dataset = load_eval_dataset()

    assert len(dataset) == 10

    store = ingest_documents()

    pipeline = RAGPipeline(
        store=store,
        llm=FakeLLM(),
        top_k=2,
    )

    hit_rates = []
    reciprocal_ranks = []

    for case in dataset:
        result = pipeline.answer_with_citations(
            case["question"]
        )

        retrieved = [
            citation["chunk_id"]
            for citation in result["citations"]
        ]

        # print(
        #     f"\nQuestion: {case['question']}"
        # )
        # print(
        #     f"Expected: {case['relevant_chunk_ids']}"
        # )
        # print(
        #     f"Retrieved: {retrieved}"
        # )

        hit_rates.append(
            hit_rate(
                retrieved,
                case["relevant_chunk_ids"],
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved,
                case["relevant_chunk_ids"],
            )
        )

    average_hit_rate = sum(hit_rates) / len(hit_rates)
    average_reciprocal_rank = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    assert average_hit_rate >= 0.8
    assert average_reciprocal_rank >= 0.6