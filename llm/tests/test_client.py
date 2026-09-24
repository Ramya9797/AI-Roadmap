from unittest.mock import patch

from llm.client import create_llm


def test_create_llm():
    with patch("llm.client.ChatOllama") as mock_chat_ollama:
        create_llm()

        mock_chat_ollama.assert_called_once_with(
            model="llama3.2",
            temperature=0,
        )