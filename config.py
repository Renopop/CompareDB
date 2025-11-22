"""
Configuration centralisée pour CompareDB.
"""

import os

# =========================
#  Configuration API (mode en ligne)
# =========================

# Embeddings API
SNOWFLAKE_API_BASE = os.getenv(
    "SNOWFLAKE_API_BASE",
    "https://api.dev.dassault-aviation.pro/snowflake-arctic-embed-l-v2.0/v1"
)
SNOWFLAKE_API_KEY = os.getenv("SNOWFLAKE_API_KEY", "token")

# LLM API
DALLEM_API_BASE = os.getenv(
    "DALLEM_API_BASE",
    "https://api.dev.dassault-aviation.pro/dallem-pilote/v1"
)
DALLEM_API_KEY = os.getenv("DALLEM_API_KEY", "EMPTY")

# Reranker API (optionnel)
RERANK_API_BASE = os.getenv(
    "RERANK_API_BASE",
    "https://api.dev.dassault-aviation.pro/bge-reranker-v2-m3/v1/"
)
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "EMPTY")

# SSL
DISABLE_SSL_VERIFY = os.getenv("DISABLE_SSL_VERIFY", "true")
VERIFY_SSL = not (DISABLE_SSL_VERIFY.lower() in ("1", "true", "yes", "on"))

# =========================
#  Configuration modèles locaux (mode hors ligne)
# =========================

# Modèles LLM disponibles
AVAILABLE_LLM_MODELS = {
    "qwen": {
        "name": "Qwen 2.5 3B Instruct",
        "path": "D:\\IA Test\\models\\Qwen\\Qwen2.5-3B-Instruct",
        "description": "Modèle léger et rapide pour l'analyse de texte",
    },
    "mistral": {
        "name": "Mistral 7B Instruct v0.3",
        "path": "D:\\IA Test\\models\\mistralai\\Mistral-7B-Instruct-v0.3",
        "description": "Modèle performant pour des analyses complexes",
    },
}

# Modèles d'embedding
AVAILABLE_EMBEDDING_MODELS = {
    "bge-m3": {
        "name": "BGE-M3",
        "path": "D:\\IA Test\\models\\BAAI\\bge-m3",
        "description": "Modèle multilingue de haute qualité",
    },
}

# Modèle par défaut
DEFAULT_LLM_MODEL = "qwen"
DEFAULT_EMBEDDING_MODEL = "bge-m3"

# =========================
#  Configuration application
# =========================

# Paramètres par défaut
DEFAULT_THRESHOLD = 0.78
DEFAULT_BATCH_SIZE = 16
DEFAULT_MATCH_MODE = "full"
DEFAULT_TOPK = 10
DEFAULT_LLM_MAX = 200

# Limites
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_ROWS = 100000

# Répertoires
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Créer les répertoires si nécessaires
for directory in [OUTPUT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# =========================
#  Configuration logging
# =========================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG_MODE = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes", "on")

# =========================
#  Configuration Flask
# =========================

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "1").lower() in ("1", "true", "yes", "on")
