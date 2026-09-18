from app.agent import run_agent


def test_max_tool_steps():

    result = run_agent()

    assert result == 3