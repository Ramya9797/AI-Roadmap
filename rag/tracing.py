class TraceRecorder:
    def __init__(self):
        self.trace = {}

    def record(self, key, value):
        self.trace[key] = value

    def get_trace(self):
        return self.trace.copy()