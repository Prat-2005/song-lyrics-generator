import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# DATA_PATH points at the directory of artist lyric CSVs — this IS the
# dataset. There used to also be a separate cache/ directory of one-JSON-
# per-song files that the retriever read from instead; that's gone. The
# retriever now reads these CSVs directly, and scripts/fetch_lyrics.py
# appends newly-fetched songs straight into them.
DATA_PATH = os.getenv("DATA_PATH", str(DATA_DIR))
if not Path(DATA_PATH).is_absolute():
    DATA_PATH = str(BASE_DIR / DATA_PATH)

INDEX_PATH = os.getenv("INDEX_PATH", str(MODELS_DIR / "faiss_index.bin"))
if not Path(INDEX_PATH).is_absolute():
    INDEX_PATH = str(BASE_DIR / INDEX_PATH)

METADATA_PATH = os.getenv("METADATA_PATH", str(MODELS_DIR / "metadata.pkl"))
if not Path(METADATA_PATH).is_absolute():
    METADATA_PATH = str(BASE_DIR / METADATA_PATH)

# --- LLM Configuration ---
LOCAL_PROVIDER = os.getenv("LOCAL_PROVIDER", "ollama")
LOCAL_URL = os.getenv("LOCAL_URL", "http://localhost:11434/api/chat")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.2:3b")
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "120"))

# --- Fallback LLM Configuration ---
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "groq")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "llama-3.1-8b-instant")
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "")
FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", "https://api.groq.com/openai/v1")

# --- Retrieval Settings ---
RETRIEVAL_MODEL_NAME = os.getenv("RETRIEVAL_MODEL_NAME", "all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HF_TOKEN", "")


def validate_config():
    """Basic check to ensure critical paths exist or are configured."""
    missing = []
    if not os.path.exists(DATA_PATH):
        missing.append(f"DATA_PATH: {DATA_PATH}")
    elif os.path.isdir(DATA_PATH):
        csv_files = [name for name in os.listdir(DATA_PATH) if name.lower().endswith('.csv')]
        if not csv_files:
            missing.append(f"CSV files in DATA_PATH directory: {DATA_PATH}")

    if missing:
        print("Configuration Warning: The following required files were not found:")
        for m in missing:
            print(f"  - {m}")
        print("Please check your .env file or directory structure.\n")
    else:
        print("Configuration validated.")
