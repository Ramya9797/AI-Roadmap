
class ConstrainedAgent:
    def __init__(
        self,
        search_documents=None,
        get_document=None,
        list_documents=None,
        trace_store=None,
    ):
        self.allowed_actions = [
            "search_documents",
            "get_document",
            "list_documents",
        ]

        self.search_documents = search_documents
        self.get_document = get_document
        self.list_documents = list_documents
        self.trace_store = trace_store

    def _record_trace(
        self,
        action,
        arguments,
        result,
        token_usage=None,
    ):
        if self.trace_store is not None:
            self.trace_store.record(
                action=action,
                arguments=arguments,
                result=result,
                token_usage=token_usage,
            )

    def execute(self, action, token_usage=None, **kwargs):
        if not action.strip():
            raise ValueError("Action cannot be empty")

        if action not in self.allowed_actions:
            raise ValueError(
                f"Action not allowed: {action}"
            )

        if action == "search_documents":
            query = kwargs.get("query")

            if not query or not query.strip():
                raise ValueError(
                    "query is required for search_documents"
                )

            if self.search_documents is None:
                result = {
                    "action": action,
                    "query": query,
                }

                self._record_trace(
                    action,
                    {"query": query},
                    result,
                    token_usage,
                )

                return result

            results = self.search_documents(query)

            result = {
                "action": action,
                "query": query,
                "results": results,
            }

            self._record_trace(
                action,
                {"query": query},
                result,
                token_usage,
            )

            return result

        if action == "get_document":
            document_id = kwargs.get("document_id")

            if not document_id or not document_id.strip():
                raise ValueError(
                    "document_id is required for get_document"
                )

            if self.get_document is None:
                result = {
                    "action": action,
                    "document_id": document_id,
                }

                self._record_trace(
                    action,
                    {"document_id": document_id},
                    result,
                    token_usage,
                )

                return result

            document = self.get_document(document_id)

            result = {
                "action": action,
                "document_id": document_id,
                "document": document,
            }

            self._record_trace(
                action,
                {"document_id": document_id},
                result,
                token_usage,
            )

            return result

        if action == "list_documents":
            if kwargs:
                raise ValueError(
                    "list_documents does not accept arguments"
                )

            if self.list_documents is None:
                result = {
                    "action": action,
                    "documents": [],
                }

                self._record_trace(
                    action,
                    {},
                    result,
                    token_usage,
                )

                return result

            documents = self.list_documents()

            result = {
                "action": action,
                "documents": documents,
            }

            self._record_trace(
                action,
                {},
                result,
                token_usage,
            )

            return result

        raise ValueError(
            f"Action not allowed: {action}"
        )
