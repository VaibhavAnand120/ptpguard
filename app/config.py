import os
from dotenv import load_dotenv

load_dotenv()

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "rules").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Evidence Dimensions (Sign convention: POSITIVE = supports credibility, NEGATIVE = reduces credibility)
DIMENSIONS = [
    "commitment",           # positive = firm commitment, negative = refusal / weak commitment
    "specificity",          # positive = clear amount/date/action, negative = vague commitment
    "borrower_initiation",  # positive = borrower voluntarily proposes PTP, negative = agent is driving
    "confirmation",         # positive = borrower explicitly confirms, negative = avoids / does not confirm
    "feasibility",          # positive = evidence borrower can realistically pay, negative = inability
    "conditionality",       # positive = unconditional commitment, negative = conditional ("if salary comes", etc.)
    "hardship",             # positive = no meaningful hardship barrier, negative = genuine financial hardship
    "agent_pressure",       # positive = borrower-originated / low pressure, negative = strong agent pressure
    "third_party",          # positive = borrower confirms responsibility, negative = third party promise / interference
    "escape_signal",        # positive = firm commitment, negative = avoidance / escape language
]

# Configurable Weights (Must sum to 1.00)
DEFAULT_WEIGHTS = {
    "commitment": 0.16,
    "specificity": 0.18,
    "borrower_initiation": 0.14,
    "confirmation": 0.16,
    "feasibility": 0.12,
    "conditionality": 0.10,
    "hardship": 0.06,
    "agent_pressure": 0.04,
    "third_party": 0.02,
    "escape_signal": 0.02,
}

# Soft Saturation Scale (raw_state / SCALE into tanh)
# e.g., raw_state = 2.0 -> tanh(2/2.5) = 0.66, raw_state = 5.0 -> tanh(5/2.5) = 0.96
DEFAULT_SCALE = float(os.getenv("EVIDENCE_SCALE", "2.5"))

# Classification Thresholds (on normalized_state in [-1, +1])
THRESHOLDS = {
    "strong_negative": -0.30,
    "moderate_negative": -0.15,
    "positive": 0.05,
    "strong_positive": 0.25,
}
