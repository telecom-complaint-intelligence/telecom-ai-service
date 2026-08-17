"""
config.py

Central configuration for our Agentic AI application.
"""

import os

from dotenv import load_dotenv


# Load variables from .env

load_dotenv()


# ------------------------------------------------------------
# Qwen configuration
# ------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("QWEN_API_KEY")

QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-72B-Instruct")


# ------------------------------------------------------------
# Agent & Guardrail configuration
# ------------------------------------------------------------

MAX_REPLANS = int(os.getenv("MAX_REPLANS", "1"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
CRITICAL_CONFIDENCE_THRESHOLD = float(os.getenv("CRITICAL_CONFIDENCE_THRESHOLD", "0.75"))
ENABLE_EXECUTION = os.getenv("ENABLE_EXECUTION", "True").lower() in ("true", "1", "yes")
STRICT_POLICY_ENFORCEMENT = os.getenv("STRICT_POLICY_ENFORCEMENT", "True").lower() in ("true", "1", "yes")
MAX_COMPLAINT_TEXT_LENGTH = int(os.getenv("MAX_COMPLAINT_TEXT_LENGTH", "10000"))
MIN_COMPLAINT_TEXT_LENGTH = int(os.getenv("MIN_COMPLAINT_TEXT_LENGTH", "3"))