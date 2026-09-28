from rag.version import (
    APP_VERSION,
    MODEL_VERSION,
    PROMPT_VERSION,
    INDEX_VERSION,
)


def test_app_version_is_defined():
    assert APP_VERSION == "1.0.0"


def test_model_version_is_defined():
    assert MODEL_VERSION == "all-MiniLM-L6-v2"


def test_prompt_version_is_defined():
    assert PROMPT_VERSION == "1.0.0"


def test_index_version_is_defined():
    assert INDEX_VERSION == "1.0.0"