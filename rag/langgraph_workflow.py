from typing import Any, Callable, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from rag.human_approval import HumanApproval


class AgentState(TypedDict, total=False):
    action: str
    query: str
    results: List[Dict[str, Any]]
    error: str
    status: str


class LangGraphAgent:
    """
    LangGraph-based workflow with human approval
    before document actions are executed.
    """

    def __init__(
        self,
        search_documents: Callable[[str], List[Dict[str, Any]]],
    ):
        self.search_documents = search_documents
        self.approval = HumanApproval()

        graph = StateGraph(AgentState)

        graph.add_node(
            "validate",
            self._validate,
        )

        graph.add_node(
            "approval",
            self._approval,
        )

        graph.add_node(
            "search_documents",
            self._search_documents,
        )

        graph.add_edge(
            START,
            "validate",
        )

        graph.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {
                "approval": "approval",
                "error": END,
            },
        )

        graph.add_conditional_edges(
            "approval",
            self._route_after_approval,
            {
                "search_documents": "search_documents",
                "end": END,
            },
        )

        graph.add_edge(
            "search_documents",
            END,
        )

        self.graph = graph.compile()

    def _validate(
        self,
        state: AgentState,
    ) -> AgentState:
        action = state.get(
            "action",
            "",
        ).strip()

        if action != "search_documents":
            return {
                **state,
                "error": f"Unknown action: {action}",
            }

        query = state.get(
            "query",
            "",
        ).strip()

        if not query:
            return {
                **state,
                "error": "query is required",
            }

        return {
            **state,
            "action": action,
            "query": query,
        }

    def _route_after_validation(
        self,
        state: AgentState,
    ) -> str:
        if state.get("error"):
            return "error"

        return "approval"

    def _approval(
        self,
        state: AgentState,
    ) -> AgentState:
        return {
            **state,
            "status": self.approval.status,
        }

    def _route_after_approval(
        self,
        state: AgentState,
    ) -> str:
        if state.get("status") == "approved":
            return "search_documents"

        return "end"

    def _search_documents(
        self,
        state: AgentState,
    ) -> AgentState:
        results = self.search_documents(
            state["query"]
        )

        return {
            **state,
            "results": results,
        }

    def approve(self):
        self.approval.approve()

    def reject(self):
        self.approval.reject()

    def run(
        self,
        action: str,
        query: str = "",
    ) -> Dict[str, Any]:
        result = self.graph.invoke(
            {
                "action": action,
                "query": query,
            }
        )

        if result.get("error"):
            raise ValueError(
                result["error"]
            )

        return result