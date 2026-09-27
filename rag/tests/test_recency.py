from datetime import datetime, timedelta

from rag.recency import calculate_recency_score


def test_recent_document_gets_higher_score():
    now = datetime.now()

    recent = calculate_recency_score(
        created_at=now - timedelta(days=1),
        now=now,
    )

    old = calculate_recency_score(
        created_at=now - timedelta(days=30),
        now=now,
    )

    assert recent > old


def test_same_date_gets_full_score():
    now = datetime.now()

    score = calculate_recency_score(
        created_at=now,
        now=now,
    )

    assert score == 1.0


def test_recency_score_is_between_zero_and_one():
    now = datetime.now()

    score = calculate_recency_score(
        created_at=now - timedelta(days=100),
        now=now,
    )

    assert 0.0 <= score <= 1.0


def test_future_date_does_not_break_score():
    now = datetime.now()

    score = calculate_recency_score(
        created_at=now + timedelta(days=5),
        now=now,
    )

    assert 0.0 <= score <= 1.0