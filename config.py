import os
from pathlib import Path
from dotenv import load_dotenv

# Directories
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env explicitly
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
BILSTM_DIR = MODELS_DIR / "bilstm"
BERT_DIR = MODELS_DIR / "bert"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they do not exist
for directory in [DATA_DIR, MODELS_DIR, BILSTM_DIR, BERT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database Configuration
DEFAULT_DB_PATH = str(DATA_DIR / "emotion_platform.db")
DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    DB_PATH = DEFAULT_DB_PATH

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Default Model Selection ('bert' or 'bilstm')
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "bert").lower()

# Emotion classes mapping and colors
EMOTION_CLASSES = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

EMOTION_COLORS = {
    "Bored": "#9E9E9E",      # Grey
    "Confident": "#4CAF50",  # Green
    "Confused": "#FF9800",   # Orange
    "Curious": "#2196F3",    # Blue
    "Frustrated": "#F44336"  # Red
}

EMOTION_EMOJIS = {
    "Bored": "😴",
    "Confident": "😎",
    "Confused": "😕",
    "Curious": "🤔",
    "Frustrated": "😩"
}
