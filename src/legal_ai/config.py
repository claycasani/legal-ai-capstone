from pathlib import Path

APP_TITLE = "Contract Clarity"
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DB_DIR = DATA_DIR / "uploaded_chroma"

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4.1-mini"

DEFAULT_LOCAL_GENERATION_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_LORA_ADAPTER_PATH = "models/contract-clarity-lora"

SUMMARY_QUERIES = [
    "What is this contract about?",
    "What are the main obligations in this agreement?",
    "What is the exact rent amount payment amount fee amount security deposit due date late fee?",
    "How much rent does the tenant pay and when is rent due?",
    "What deposits fees charges utilities or late penalties does the signer owe?",
    "How can the agreement be terminated?",
    "What risks or limitations should a signer notice?",
    "What questions should the signer clarify before signing?",
]

CLAUSE_QUERIES = {
    "Payment": "rent monthly rent amount payment due date security deposit late fee utilities charges fees",
    "Termination": "termination cancellation expiration renewal breach notice vacate early termination default",
    "Liability": "liability indemnification damages losses limitation of liability responsible for damage",
    "Confidentiality": "confidentiality non-disclosure proprietary information",
    "IP Ownership": "intellectual property ownership license copyright patent work product",
    "Governing Law": "governing law jurisdiction venue court arbitration dispute",
}
