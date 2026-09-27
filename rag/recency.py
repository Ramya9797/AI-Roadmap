from datetime import datetime
import math


def calculate_recency_score(
    created_at: datetime,
    now: datetime | None = None,
) -> float:
    """
    Calculate a recency score between 0 and 1.

    Newer documents receive a higher score.
    Older documents receive a lower score.
    """

    if now is None:
        now = datetime.now()

    age_seconds = max(0, (now - created_at).total_seconds())
    age_days = age_seconds / 86400

    score = math.exp(-age_days / 30)

    return max(0.0, min(1.0, score))