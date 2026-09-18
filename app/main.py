# import io
# import logging

# import pandas as pd
# from fastapi import FastAPI, File, HTTPException, UploadFile

# from app.config import ENVIRONMENT, LOG_LEVEL
# from app.schemas import HealthResponse, CSVAnalysisResponse


# logging.basicConfig(
#     level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# logger = logging.getLogger(__name__)


# app = FastAPI(
#     title="CSV Insights API",
#     description="AI Engineer Roadmap - Week 1",
#     version="1.0.0"
# )


# @app.get("/")
# def root():
#     return {
#         "message": "CSV Insights API is running",
#         "environment": ENVIRONMENT
#     }


# @app.get("/health", response_model=HealthResponse)
# def health():
#     return HealthResponse(
#         status="healthy"
#     )


# @app.get("/ready")
# def readiness():
#     return {
#         "status": "ready"
#     }


# @app.post(
#     "/api/v1/analyze",
#     response_model=CSVAnalysisResponse
# )
# async def analyze_csv(
#     file: UploadFile = File(...)
# ):

#     if not file.filename:
#         raise HTTPException(
#             status_code=400,
#             detail="Filename is required"
#         )

#     if not file.filename.lower().endswith(".csv"):
#         raise HTTPException(
#             status_code=400,
#             detail="Only CSV files are supported"
#         )

#     logger.info(
#         "Received CSV file: %s",
#         file.filename
#     )

#     contents = await file.read()

#     if not contents:
#         raise HTTPException(
#             status_code=400,
#             detail="Uploaded file is empty"
#         )

#     try:
#         df = pd.read_csv(
#             io.BytesIO(contents)
#         )

#     except Exception as exc:
#         logger.error(
#             "Failed to read CSV: %s",
#             exc
#         )

#         raise HTTPException(
#             status_code=400,
#             detail="Invalid CSV file"
#         )

#     if df.empty:
#         raise HTTPException(
#             status_code=400,
#             detail="CSV contains no data rows"
#         )

#     logger.info(
#         "CSV loaded successfully: %s rows, %s columns",
#         len(df),
#         len(df.columns)
#     )

#     missing_values = (
#         df.isnull()
#         .sum()
#         .astype(int)
#         .to_dict()
#     )

#     missing_percentages = (
#         (df.isnull().mean() * 100)
#         .round(2)
#         .to_dict()
#     )

#     data_types = {
#         column: str(dtype)
#         for column, dtype in df.dtypes.items()
#     }

#     unique_values = {
#         column: int(df[column].nunique())
#         for column in df.columns
#     }

#     numeric_df = df.select_dtypes(
#         include="number"
#     )

#     numeric_summary = {}

#     for column in numeric_df.columns:
#         numeric_summary[column] = {
#             "mean": float(
#                 numeric_df[column].mean()
#             ),
#             "min": float(
#                 numeric_df[column].min()
#             ),
#             "max": float(
#                 numeric_df[column].max()
#             ),
#             "median": float(
#                 numeric_df[column].median()
#             )
#         }

#     logger.info(
#         "Analysis completed for: %s",
#         file.filename
#     )

#     return CSVAnalysisResponse(
#         filename=file.filename,
#         rows=len(df),
#         columns=len(df.columns),
#         column_names=df.columns.tolist(),
#         missing_values=missing_values,
#         missing_percentages=missing_percentages,
#         data_types=data_types,
#         unique_values=unique_values,
#         numeric_summary=numeric_summary
#     )

import io
import time
import logging

import joblib
import pandas as pd

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import ENVIRONMENT, LOG_LEVEL
from app.schemas import HealthResponse, CSVAnalysisResponse, CopilotResponse

from app.llm import MockLLM


# -----------------------------
# Logging
# -----------------------------

# logging.basicConfig(level=logging.INFO)

logging.basicConfig(
    level=getattr(
        logging,
        LOG_LEVEL.upper(),
        logging.INFO
    ),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------
# Load model
# -----------------------------

model = joblib.load(
    "models/churn_model.joblib"
)


# -----------------------------
# FastAPI
# -----------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

llm = MockLLM()
# -----------------------------
# Request schema
# -----------------------------

class CustomerRequest(BaseModel):

    age: int = Field(
        ge=0
    )

    monthly_bill: float = Field(
        ge=0
    )

    tenure: int = Field(
        ge=0
    )

    support_calls: int = Field(
        ge=0
    )

    contract: str

    payment: str

    internet_service: str


# -----------------------------
# Health check
# -----------------------------



@app.get("/")
def root():
    return {
        "message": "CSV Insights API is running",
        "environment": ENVIRONMENT
    }

@app.get("/health", response_model=HealthResponse)
def health():

    return HealthResponse(
        status="healthy"
    )


@app.get("/ready")
def readiness():
    return {
        "status": "ready"
    }


@app.post(
    "/api/v1/analyze",
    response_model=CSVAnalysisResponse
)
async def analyze_csv(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    try:
        df = pd.read_csv(
            io.BytesIO(contents)
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV file"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV contains no data rows"
        )

    missing_values = (
        df.isnull()
        .sum()
        .astype(int)
        .to_dict()
    )

    missing_percentages = (
        (df.isnull().mean() * 100)
        .round(2)
        .to_dict()
    )

    data_types = {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }

    unique_values = {
        column: int(df[column].nunique())
        for column in df.columns
    }

    numeric_df = df.select_dtypes(
        include="number"
    )

    numeric_summary = {}

    for column in numeric_df.columns:
        numeric_summary[column] = {
            "mean": float(numeric_df[column].mean()),
            "min": float(numeric_df[column].min()),
            "max": float(numeric_df[column].max()),
            "median": float(numeric_df[column].median())
        }

    return CSVAnalysisResponse(
        filename=file.filename,
        rows=len(df),
        columns=len(df.columns),
        column_names=df.columns.tolist(),
        missing_values=missing_values,
        missing_percentages=missing_percentages,
        data_types=data_types,
        unique_values=unique_values,
        numeric_summary=numeric_summary
    )
# -----------------------------
# Prediction
# -----------------------------

@app.post("/predict")
def predict(customer: CustomerRequest):

    start_time = time.perf_counter()

    data = pd.DataFrame([
        {
            "age": customer.age,

            "monthly_bill":
                customer.monthly_bill,

            "tenure":
                customer.tenure,

            "support_calls":
                customer.support_calls,

            "contract":
                customer.contract,

            "payment":
                customer.payment,

            "internet_service":
                customer.internet_service
        }
    ])

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    latency = (
        time.perf_counter() - start_time
    ) * 1000

    logger.info(
        "Prediction completed: latency_ms=%.2f",
        latency
    )

    return {
        "prediction": int(prediction),

        "churn_probability":
            round(float(probability), 4),

        "latency_ms":
            round(latency, 2)
    }

@app.post(
    "/copilot",
    response_model=CopilotResponse
)
def copilot(ticket: str):

    result = llm.generate(ticket)

    validated_result = (
        CopilotResponse.model_validate(
            result
        )
    )

    return validated_result