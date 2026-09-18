# from pydantic import BaseModel


# class HealthResponse(BaseModel):
#     status: str


# class CSVAnalysisResponse(BaseModel):
#     filename: str
#     rows: int
#     columns: int
#     column_names: list[str]
#     missing_values: dict[str, int]
#     missing_percentages: dict[str, float]
#     data_types: dict[str, str]
#     unique_values: dict[str, int]
#     numeric_summary: dict[str, dict[str, float]]



from enum import Enum
from pydantic import BaseModel


class Intent(str, Enum):
    ORDER_STATUS = "order_status"
    REFUND_REQUEST = "refund_request"
    CANCEL_ORDER = "cancel_order"
    GENERAL = "general"


class Action(str, Enum):
    LOOKUP_ORDER = "lookup_order"
    CHECK_REFUND_POLICY = "check_refund_policy"
    NONE = "none"


class CopilotResponse(BaseModel):
    intent: Intent
    action: Action
    order_id: str | None = None
    reply: str
    citations: list[str]

class HealthResponse(BaseModel):
    status: str


class CSVAnalysisResponse(BaseModel):
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    missing_values: dict
    missing_percentages: dict
    data_types: dict
    unique_values: dict
    numeric_summary: dict
