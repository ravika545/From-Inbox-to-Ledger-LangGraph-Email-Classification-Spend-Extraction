from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env from CWD

DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL  = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL     = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

CONFIDENCE_THRESHOLD   = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
AMOUNT_THRESHOLD_INR   = float(os.getenv("AMOUNT_THRESHOLD_INR", "5000"))

if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY missing — add it to your .env file")