import os

from dotenv import load_dotenv

load_dotenv()


ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development"
)

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO"
)