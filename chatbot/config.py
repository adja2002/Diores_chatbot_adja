"""
Configuration du chatbot DIORES
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Configuration Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Configuration Mistral
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# Configuration ChromaDB
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "diores_formations")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chatbot/data/chroma_db")

# Configuration mémoire
MEMORY_STORAGE_PATH = os.getenv("MEMORY_STORAGE_PATH", "chatbot/data/memory")

# Configuration serveur
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

# Chemins des modèles
MODELS_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "Models"
)

