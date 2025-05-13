"""
LLM and embedding model initialization and utilities.
"""
import os
import logging
from typing import Optional, Union, Any, Dict, List

# Import HuggingFacePipeline which is always available
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings as BaseHuggingFaceEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
import torch

# Import FakeListLLM which is always available
from langchain_community.llms.fake import FakeListLLM

# Try to import LlamaCpp, but don't fail if it's not available
LLAMACPP_AVAILABLE = False
try:
    import llama_cpp
    from langchain_community.llms import LlamaCpp
    LLAMACPP_AVAILABLE = True
except ImportError:
    logging.warning("llama-cpp-python is not installed. GGUF models will not be available.")

from backend.config import (
    MODELS_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    REPETITION_PENALTY
)

logger = logging.getLogger(__name__)

def get_embedding_model(model_name: Optional[str] = None) -> Union[BaseHuggingFaceEmbeddings, FakeEmbeddings]:
    """
    Initialize and return an embedding model.
    
    Args:
        model_name: Name of the embedding model to use. If None, uses the default model.
    
    Returns:
        An initialized embedding model instance.
    """
    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    logger.info(f"Loading embedding model: {model_name}")
    
    # Models known to produce 768-dimensional embeddings
    models_768dim = [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-distilroberta-v1",
        "sentence-transformers/bert-base-nli-mean-tokens"
    ]
    
    # If the vector store was created with 768-dimensional embeddings,
    # we need to ensure we use a model that produces 768-dimensional embeddings
    if model_name not in models_768dim and not model_name.startswith("sentence-transformers/all-mpnet-base"):
        logger.warning(f"Model {model_name} may not produce 768-dimensional embeddings")
        logger.warning(f"Using all-mpnet-base-v2 instead to match vector store dimensionality")
        model_name = "sentence-transformers/all-mpnet-base-v2"
    
    try:
        # Check if model exists locally
        model_path = os.path.join(MODELS_DIR, os.path.basename(model_name))
        if os.path.exists(model_path):
            logger.info(f"Using local embedding model from: {model_path}")
            model_location = model_path
        else:
            logger.info(f"Using embedding model from HuggingFace Hub: {model_name}")
            model_location = model_name
        
        # Initialize and return the embedding model
        return BaseHuggingFaceEmbeddings(
            model_name=model_location,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        error_message = f"Error loading embedding model: {e}"
        logger.error(error_message)
        logger.error("Please ensure the required packages are installed:")
        logger.error("  - sentence-transformers (for HuggingFace embeddings)")
        logger.error("  - torch (for PyTorch support)")
        logger.error("Or use llama.cpp embeddings by setting EMBEDDING_TYPE='llama_cpp'")
        raise RuntimeError(error_message)

def get_llm(model_name: Optional[str] = None) -> Union[HuggingFacePipeline, FakeListLLM]:
    """
    Initialize and return an LLM.
    
    Args:
        model_name: Name of the LLM model to use. If None, uses the default model.
    
    Returns:
        An initialized LLM instance (either HuggingFacePipeline or FakeListLLM).
    """
    model_name = model_name or DEFAULT_LLM_MODEL
    logger.info(f"Loading LLM model: {model_name}")
    
    # Check if model exists locally, otherwise use from HuggingFace
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        logger.info(f"Using local LLM model from: {model_path}")
        model_location = model_path
        
        # Check if the model is a GGUF file
        if model_path.endswith(".gguf"):
            if LLAMACPP_AVAILABLE:
                logger.info("Loading GGUF model with LlamaCpp")
                return LlamaCpp(
                    model_path=model_path,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_NEW_TOKENS,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    repeat_penalty=REPETITION_PENALTY,
                    n_ctx=8192,  # Context window size
                    n_gpu_layers=-1,  # Use all available GPU layers
                    verbose=True,
                )
            else:
                # Raise a clear error message instead of using a mock LLM
                error_message = (
                    "GGUF model detected but llama-cpp-python is not installed. "
                    "Please install it with: pip install llama-cpp-python"
                )
                logger.error(error_message)
                raise ImportError(error_message)
    else:
        logger.info(f"Using LLM model from HuggingFace Hub: {model_name}")
        model_location = model_name
    
    # For standard Hugging Face models
    # Configure quantization if GPU is available
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using quantized model.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        logger.info("CUDA is not available. Using CPU.")
        quantization_config = None
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    model = AutoModelForCausalLM.from_pretrained(
        model_location,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Create LangChain pipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
