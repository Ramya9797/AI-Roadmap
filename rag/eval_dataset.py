from typing import List, Dict


EVAL_DATASET = [
    {
        "question": "What is the refund policy?",
        "relevant_chunk_ids": ["order_policy.md:0"],
    },
    {
        "question": "How can I check my order status?",
        "relevant_chunk_ids": ["customer_support.md:0"],
    },
    {
        "question": "What is the company's strategy?",
        "relevant_chunk_ids": ["churn_strategy.md:0"],
    },
    {
        "question": "How long does a refund take?",
        "relevant_chunk_ids": ["order_policy.md:0"],
    },
    {
        "question": "What should I do about my order?",
        "relevant_chunk_ids": ["customer_support.md:0"],
    },
]


def load_eval_dataset() -> List[Dict]:
    return EVAL_DATASET.copy()