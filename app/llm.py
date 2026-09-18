class MockLLM:

    def generate(self, ticket: str):
        ticket_lower = ticket.lower()

        if (
            "order" in ticket_lower
            or "package" in ticket_lower
            or "shipment" in ticket_lower
        ) and (
            "where" in ticket_lower
            or "arrived" in ticket_lower
            or "arrive" in ticket_lower
            or "status" in ticket_lower
            or "track" in ticket_lower
        ):
            return {
                "intent": "order_status",
                "action": "lookup_order",
                "order_id": "1001",
                "reply": "I found your order. Let me check the current status.",
                "citations": ["order_lookup"]
            }


        # if "order" in ticket_lower and (
        #     "where" in ticket_lower
        #     or "arrived" in ticket_lower
        #     or "status" in ticket_lower
        # ):
        #     return {
        #         "intent": "order_status",
        #         "action": "lookup_order",
        #         "order_id": "1001"
        #     }

        if (
            "refund" in ticket_lower
            or "money back" in ticket_lower
            or "return" in ticket_lower
        ):
            return {
                "intent": "refund_request",
                "action": "check_refund_policy",
                "order_id": None,
                "reply": "I can help you with the refund policy.",
                "citations": ["refund_policy"]
            }
        # if "refund" in ticket_lower or "money back" in ticket_lower:
        #     return {
        #         "intent": "refund_request",
        #         "action": "check_refund_policy",
        #         "order_id": None
        #     }


        return {
            "intent": "general",
            "action": "none",
            "order_id": None,
            "reply": "How can I help you today?",
            "citations": []
        }
        # return {
        #     "intent": "general",
        #     "action": "none",
        #     "order_id": None
        # }



