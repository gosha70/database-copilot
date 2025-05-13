"""
Configuration settings for the Database Copilot application.
"""
import os
from pathlib import Path
import streamlit as st

# Base directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, "data")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")

# Model settings
MODELS_DIR = os.path.join(DATA_DIR, "hf_models")
DEFAULT_LLM_MODEL = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # This should be the file name in MODELS_DIR
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# LLM type (local, openai, claude, gemini, mistral, deepseek)
# First check streamlit secrets, then environment variables
if hasattr(st, 'secrets') and 'LLM_TYPE' in st.secrets:
    LLM_TYPE = st.secrets['LLM_TYPE']
else:
    LLM_TYPE = os.environ.get("LLM_TYPE", "local")

# API keys for external LLMs
# First check streamlit secrets, then environment variables
if hasattr(st, 'secrets'):
    # OpenAI
    if 'OPENAI_API_KEY' in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
    
    # Claude
    if 'ANTHROPIC_API_KEY' in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets['ANTHROPIC_API_KEY']
    
    # Gemini
    if 'GOOGLE_API_KEY' in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']
    
    # Mistral
    if 'MISTRAL_API_KEY' in st.secrets:
        os.environ["MISTRAL_API_KEY"] = st.secrets['MISTRAL_API_KEY']
    
    # DeepSeek
    if 'DEEPSEEK_API_KEY' in st.secrets:
        os.environ["DEEPSEEK_API_KEY"] = st.secrets['DEEPSEEK_API_KEY']

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
NUM_RETRIEVAL_RESULTS = 8

# Logging
LOG_LEVEL = "INFO"
