import pytest
from pydantic import ValidationError
from app.schemas import CopilotResponse


def test_valid_response():

    response = CopilotResponse(
        intent="order_status",
        action="lookup_order",
        order_id="1001",
        reply="Your order is shipped.",
        citations=["order_lookup"]
    )

    assert response.intent == "order_status"
    assert response.action == "lookup_order"
    assert response.order_id == "1001"


def test_invalid_response():

    with pytest.raises(ValidationError):

        CopilotResponse(
            intent="order_status"
        )    

def test_invalid_intent():

    with pytest.raises(ValidationError):

        CopilotResponse(
            intent="pizza_delivery",
            action="lookup_order",
            order_id="1001",
            reply="Your order is shipped.",
            citations=[]
        )