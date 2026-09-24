
import math
import re
from collections import Counter


class KeywordRetriever:
    def __init__(self, documents):
        self.documents = documents

        self.tokenized_documents = [
            self._tokenize(document["text"])
            for document in documents
        ]

        self.document_frequencies = Counter()

        for tokens in self.tokenized_documents:
            for token in set(tokens):
                self.document_frequencies[token] += 1

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def _matches_where(self, document, where):
        if where is None:
            return True

        metadata = document.get("metadata", {})

        for key, expected_value in where.items():
            if metadata.get(key) != expected_value:
                return False

        return True

    def search(self, query: str, top_k: int = 3, where=None):
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        total_documents = len(self.documents)
        results = []

        for index, document in enumerate(self.documents):

            # Apply metadata filter before calculating keyword score
            if not self._matches_where(document, where):
                continue

            document_tokens = self.tokenized_documents[index]

            if not document_tokens:
                continue

            token_counts = Counter(document_tokens)
            score = 0.0

            for token in query_tokens:
                term_frequency = token_counts.get(token, 0)

                if term_frequency == 0:
                    continue

                document_frequency = self.document_frequencies[token]

                idf = math.log(
                    (total_documents + 1)
                    / (document_frequency + 1)
                ) + 1

                score += term_frequency * idf

            if score > 0:
                result = {
                    "id": document["id"],
                    "text": document["text"],
                    "score": score,
                }

                if "metadata" in document:
                    result["metadata"] = document["metadata"]

                results.append(result)

        results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return results[:top_k]
