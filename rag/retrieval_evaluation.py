def hit_rate(retrieved, relevant):
    relevant = set(relevant)

    if not retrieved:
        return 0.0

    return 1.0 if any(doc in relevant for doc in retrieved) else 0.0


def reciprocal_rank(retrieved, relevant):
    relevant = set(relevant)

    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank

    return 0.0