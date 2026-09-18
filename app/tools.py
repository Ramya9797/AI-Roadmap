# Tool 1 — Order lookup

ORDERS = {
    "1001": {
        "status": "shipped",
        "delivery_date": "2026-09-20"
    },
    "1002": {
        "status": "delivered",
        "delivery_date": "2026-09-15"
    },
    "1003": {
        "status": "cancelled",
        "delivery_date": None
    }
}


def lookup_order(order_id: str):
    return ORDERS.get(order_id)


# Tool 2 — Refund policy

def check_refund_policy():
    return {
        "eligible_days": 30,
        "condition": "Product must be unused",
        "refund_method": "Original payment method"
    }