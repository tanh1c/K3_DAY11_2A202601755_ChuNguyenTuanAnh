"""
Lab 11 — Configuration & API Key Setup
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDENT_ID = "2A202601755"


def create_adk_model() -> LiteLlm:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    return LiteLlm(model=f"openai/{model}")


def setup_api_key(required: bool = True) -> bool:
    """Load the configured OpenAI key without prompting or printing it."""
    load_dotenv(REPO_ROOT / ".env")
    key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if required and not key_present:
        raise RuntimeError(
            "OPENAI_API_KEY is required for live parts. Configure it in .env."
        )
    return key_present


# Allowed banking topics (used by topic_filter)
ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

# Blocked topics (immediate reject)
BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]
