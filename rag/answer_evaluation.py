import re


def evaluate_answer(answer: str, context: str) -> dict:
    if not answer.strip():
        raise ValueError("Answer cannot be empty")

    if not context.strip():
        return {
            "supported": False,
        }

    answer_lower = answer.lower()
    context_lower = context.lower()

    answer_numbers = re.findall(r"\b\d+\b", answer_lower)
    context_numbers = re.findall(r"\b\d+\b", context_lower)

    if answer_numbers:
        for number in answer_numbers:
            if number not in context_numbers:
                return {
                    "supported": False,
                }

    answer_words = set(
        word.strip(".,!?;:()[]{}")
        for word in answer_lower.split()
    )

    context_words = set(
        word.strip(".,!?;:()[]{}")
        for word in context_lower.split()
    )

    meaningful_words = {
        word
        for word in answer_words
        if len(word) > 3
    }

    overlap = meaningful_words.intersection(context_words)

    supported = len(overlap) >= 2

    return {
        "supported": supported,
    }