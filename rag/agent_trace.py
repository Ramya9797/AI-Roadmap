import json
from pathlib import Path


class AgentTraceStore:
    def __init__(self, trace_file):
        self.trace_file = Path(trace_file)

        if self.trace_file.exists():
            with open(
                self.trace_file,
                "r",
                encoding="utf-8",
            ) as file:
                self.traces = json.load(file)
        else:
            self.traces = []

    def record(
        self,
        action,
        arguments,
        result,
        token_usage=None,
    ):
        trace = {
            "action": action,
            "arguments": arguments,
            "result": result,
        }

        if token_usage is not None:
            trace["token_usage"] = token_usage

        self.traces.append(trace)

        self.trace_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.trace_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.traces,
                file,
                indent=2,
            )

    def get_traces(self):
        return self.traces.copy()