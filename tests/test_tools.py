# import pytest

# from app.tools import lookup_order


# def failing_order_lookup(order_id: str):
#     raise Exception("Database unavailable")


# def test_safe_lookup_handles_failure():

#     result = safe_lookup_order("1001")

#     assert result["success"] is False
#     assert "Database unavailable" in result["error"]

# def test_tool_failure():
#     with pytest.raises(Exception):
#         failing_order_lookup("1001")


# def test_invalid_order():
#     result = lookup_order("9999")

#     assert result is None


# def safe_lookup_order(order_id: str):

#     try:
#         result = lookup_order(order_id)

#         return {
#             "success": True,
#             "data": result
#         }

#     except Exception as e:

#         return {
#             "success": False,
#             "error": str(e)
#         }

import pytest

from app.tools import lookup_order


def failing_order_lookup(order_id: str):
    raise Exception("Database unavailable")


def safe_lookup_order(order_id: str):

    try:
        result = failing_order_lookup(order_id)

        return {
            "success": True,
            "data": result
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


def test_safe_lookup_handles_failure():

    result = safe_lookup_order("1001")

    assert result["success"] is False
    assert "Database unavailable" in result["error"]


def test_tool_failure():

    with pytest.raises(Exception):
        failing_order_lookup("1001")


def test_invalid_order():

    result = lookup_order("9999")

    assert result is None