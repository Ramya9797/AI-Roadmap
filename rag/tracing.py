from rag.version import (
    APP_VERSION,
    MODEL_VERSION,
    PROMPT_VERSION,
    INDEX_VERSION,
)


class TraceRecorder:
    def __init__(self):
        self.trace = {
            "app_version": APP_VERSION,
            "model_version": MODEL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "index_version": INDEX_VERSION,
        }

    def record(self, key, value):
        self.trace[key] = value

    def get_trace(self):
        return self.trace.copy()