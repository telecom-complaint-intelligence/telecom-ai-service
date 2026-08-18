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
# Agent configuration
# ------------------------------------------------------------

MAX_REPLANS = 1