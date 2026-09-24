
class HybridRetriever:
    def __init__(
        self,
        semantic_retriever,
        keyword_retriever,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        exact_match_boost: float = 0.2,
    ):
        if semantic_weight < 0:
            raise ValueError("semantic_weight cannot be negative")

        if keyword_weight < 0:
            raise ValueError("keyword_weight cannot be negative")

        if semantic_weight + keyword_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than 0"
            )

        if exact_match_boost < 0:
            raise ValueError(
                "exact_match_boost cannot be negative"
            )

        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.exact_match_boost = exact_match_boost

    def search(
        self,
        query: str,
        top_k: int = 3,
        where=None,
    ):
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # -----------------------------
        # 1. Semantic Search
        # -----------------------------
        semantic_results = self.semantic_retriever.search(
            query,
            top_k=top_k,
            where=where,
        )

        # -----------------------------
        # 2. Keyword Search
        # -----------------------------
        keyword_results = self.keyword_retriever.search(
            query,
            top_k=top_k,
            where=where,
        )

        # -----------------------------
        # 3. Combine Results
        # -----------------------------
        combined = {}

        for result in semantic_results:
            result_id = result["id"]

            combined[result_id] = {
                **result,
                "semantic_score": result["score"],
                "keyword_score": 0.0,
            }

        for result in keyword_results:
            result_id = result["id"]

            if result_id not in combined:
                combined[result_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "keyword_score": result["score"],
                }
            else:
                combined[result_id]["keyword_score"] = result["score"]

        # -----------------------------
        # 4. Exact Match Boost
        # -----------------------------
        query_lower = query.lower()

        for result in combined.values():
            text = result["text"].lower()

            exact_match = query_lower in text

            result["exact_match"] = exact_match

            # -----------------------------
            # 5. Combined Score
            # -----------------------------
            result["combined_score"] = (
                self.semantic_weight * result["semantic_score"]
                + self.keyword_weight * result["keyword_score"]
            )

            if exact_match:
                result["combined_score"] += self.exact_match_boost

        # -----------------------------
        # 6. Sort by Final Score
        # -----------------------------
        ranked_results = sorted(
            combined.values(),
            key=lambda result: result["combined_score"],
            reverse=True,
        )

        # -----------------------------
        # 7. Return Top-K
        # -----------------------------
        return ranked_results[:top_k]
