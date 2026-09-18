import time

from app.timeout_demo import slow_llm


def test_slow_operation():

    start = time.time()

    result = slow_llm()

    elapsed = time.time() - start

    assert result == "response"
    assert elapsed >= 10