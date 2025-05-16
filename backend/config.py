"""
Configuration settings for the Database Copilot application.
"""
import os
from pathlib import Path

# Try to import streamlit, but provide a fallback if it's not available
try:
    import streamlit as st
except ImportError:
    # Create a dummy st object with a secrets attribute that's an empty dict
    class DummySecrets:
        def __init__(self):
            self.secrets = {}
    
    st = DummySecrets()

# Base directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, "data")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")

# Model settings
MODELS_DIR = os.path.join(DATA_DIR, "hf_models")
DEFAULT_LLM_MODEL = "/Users/george.ivan/repo/database-copilot/data/hf_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # This should be the file name in MODELS_DIR
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_LLAMA_CPP_EMBEDDING_MODEL = "nomic-embed-text-v1.5.Q4_K_M.gguf"

# Embedding type (sentence_transformers, llama_cpp, tensorflow, tfidf)
# First check streamlit secrets, then environment variables
import sys
import platform

try:
    if hasattr(st, 'secrets') and 'EMBEDDING_TYPE' in st.secrets:
        EMBEDDING_TYPE = st.secrets['EMBEDDING_TYPE']
    else:
        EMBEDDING_TYPE = os.environ.get("EMBEDDING_TYPE", "sentence_transformers")

    # On Apple Silicon (M1/M2), default to llama_cpp unless explicitly set
    if (
        EMBEDDING_TYPE == "sentence_transformers"
        and sys.platform == "darwin"
        and platform.machine() == "arm64"
        and "EMBEDDING_TYPE" not in os.environ
        and not (hasattr(st, 'secrets') and 'EMBEDDING_TYPE' in st.secrets)
    ):
        EMBEDDING_TYPE = "llama_cpp"

    # Check if we're running in torch-free mode
    if os.environ.get("LAZY_LOAD_MODELS") == "1":
        # Force llama_cpp embeddings in torch-free mode
        EMBEDDING_TYPE = "llama_cpp"
except Exception:
    # Default to sentence_transformers if there's an error with secrets
    EMBEDDING_TYPE = "sentence_transformers"

# Try to import sentence_transformers to check if it's available
try:
    import sentence_transformers
except ImportError:
    # If sentence_transformers is not available, force llama_cpp embeddings
    EMBEDDING_TYPE = "llama_cpp"
    print("sentence_transformers not available, using llama_cpp embeddings instead")

# LLM type (local, openai, claude, gemini, mistral, deepseek)
# Define as a function to always get the current value
def get_llm_type():
    """
    Get the current LLM type from environment variables or streamlit secrets.
    This function is called every time LLM_TYPE is accessed, ensuring we always
    get the most up-to-date value.
    
    Returns:
        The current LLM type (local, openai, claude, gemini, mistral, deepseek)
    """
    try:
        if hasattr(st, 'secrets') and 'LLM_TYPE' in st.secrets:
            return st.secrets['LLM_TYPE']
        else:
            return os.environ.get("LLM_TYPE", "llama_cpp")
    except Exception:
        # Default to local if there's an error with secrets
        return "local"

# Instead of using a descriptor, let's use a function to get the current LLM type
def get_current_llm_type():
    """
    Get the current LLM type from environment variables or streamlit secrets.
    
    Returns:
        The current LLM type (local, openai, claude, gemini, mistral, deepseek)
    """
    try:
        if hasattr(st, 'secrets') and 'LLM_TYPE' in st.secrets:
            return st.secrets['LLM_TYPE']
        else:
            return os.environ.get("LLM_TYPE", "local")
    except Exception:
        # Default to local if there's an error with secrets
        return "local"

# Set LLM_TYPE as a string for initial import compatibility
LLM_TYPE = get_current_llm_type()

# API keys for external LLMs
# First check streamlit secrets, then environment variables
try:
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
except Exception:
    # Continue without secrets if there's an error
    pass

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
    "java_files": os.path.join(DOCS_DIR, "java"),  # New category for Java files
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
