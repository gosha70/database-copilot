"""
Configuration settings for the Database Copilot application.
"""
import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, "data")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")

# Model settings
MODELS_DIR = os.path.join(DATA_DIR, "hf_models")
DEFAULT_LLM_MODEL = "codellama-7b-instruct"  # This should be the folder name in MODELS_DIR
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector database settings
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_store")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Document categories
DOC_CATEGORIES = {
    "liquibase_docs": os.path.join(DOCS_DIR, "liquibase"),
    "internal_guidelines": os.path.join(DOCS_DIR, "internal"),
    "example_migrations": os.path.join(DOCS_DIR, "examples"),
    "jpa_docs": os.path.join(DOCS_DIR, "jpa"),
}

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Streamlit settings
STREAMLIT_PORT = 8501

# LLM generation settings
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.2
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.1

# RAG settings
NUM_RETRIEVAL_RESULTS = 5

# Logging
LOG_LEVEL = "INFO"
