"""
Streamlit compatibility module for the Database Copilot.

This module provides compatibility wrappers for PyTorch and other libraries
that may cause issues with Streamlit's module inspection system.
"""
import logging
import os
import sys
from typing import Optional, Any, Dict, List, Union

logger = logging.getLogger(__name__)

# Global flag to track if PyTorch has been safely imported
_TORCH_IMPORTED = False
_TRANSFORMERS_IMPORTED = False

def safe_import_torch():
    """
    Safely import PyTorch to avoid issues with Streamlit's module inspection.
    
    Returns:
        The torch module if successfully imported, None otherwise.
    """
    global _TORCH_IMPORTED
    
    if _TORCH_IMPORTED:
        import torch
        return torch
    
    try:
        # Add a custom attribute to avoid Streamlit's problematic inspection
        # This is a workaround for the "__path__._path" error
        import types
        
        # Create a dummy module to replace torch._classes temporarily
        dummy_classes = types.ModuleType("_classes")
        dummy_classes.__getattr__ = lambda attr: None
        
        # Save the original sys.modules state
        original_modules = dict(sys.modules)
        
        # Import torch with the dummy _classes module
        import torch
        if hasattr(torch, "_classes"):
            # Replace the problematic module with our dummy
            sys.modules["torch._classes"] = dummy_classes
        
        # Mark as successfully imported
        _TORCH_IMPORTED = True
        
        return torch
    
    except Exception as e:
        logger.error(f"Error safely importing PyTorch: {e}")
        # Restore original sys.modules
        sys.modules = original_modules
        return None

def safe_import_transformers():
    """
    Safely import transformers to avoid issues with Streamlit's module inspection.
    
    Returns:
        A dictionary of transformers modules if successfully imported, None otherwise.
    """
    global _TRANSFORMERS_IMPORTED
    
    if _TRANSFORMERS_IMPORTED:
        import transformers
        return {
            "AutoTokenizer": transformers.AutoTokenizer,
            "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
            "pipeline": transformers.pipeline,
            "BitsAndBytesConfig": transformers.BitsAndBytesConfig
        }
    
    # First ensure torch is safely imported
    torch = safe_import_torch()
    if torch is None:
        return None
    
    try:
        # Import transformers
        import transformers
        
        # Mark as successfully imported
        _TRANSFORMERS_IMPORTED = True
        
        return {
            "AutoTokenizer": transformers.AutoTokenizer,
            "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
            "pipeline": transformers.pipeline,
            "BitsAndBytesConfig": transformers.BitsAndBytesConfig
        }
    
    except Exception as e:
        logger.error(f"Error safely importing transformers: {e}")
        return None

def get_safe_llm(model_name: Optional[str] = None, quantization_level: str = "4bit"):
    """
    Get an LLM with Streamlit compatibility.
    
    Args:
        model_name: Name of the LLM model to use. If None, uses the default model.
        quantization_level: The quantization level to use (4bit, 8bit, or none)
        
    Returns:
        An initialized LLM instance or None if initialization fails.
    """
    from backend.config import (
        MODELS_DIR,
        DEFAULT_LLM_MODEL,
        MAX_NEW_TOKENS,
        TEMPERATURE,
        TOP_P,
        TOP_K,
        REPETITION_PENALTY
    )
    
    model_name = model_name or DEFAULT_LLM_MODEL
    logger.info(f"Loading LLM model: {model_name}")
    
    # Check if model exists locally, otherwise use from HuggingFace
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        logger.info(f"Using local LLM model from: {model_path}")
        model_location = model_path
        
        # Check if the model is a GGUF file
        if model_path.endswith(".gguf"):
            try:
                # Try to import llama-cpp-python
                import llama_cpp
                from langchain_community.llms import LlamaCpp
                
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
            except ImportError:
                logger.warning("GGUF model detected but llama-cpp-python is not installed.")
                logger.warning("Falling back to a mock LLM for testing purposes.")
                # Create a more informative mock LLM
                from langchain_community.llms.fake import FakeListLLM
                return FakeListLLM(
                    responses=[
                        "ERROR: GGUF model detected but llama-cpp-python is not installed. This is a placeholder response from a fallback system. Please install llama-cpp-python to use GGUF models.",
                        "ERROR: Missing llama-cpp-python package. This is a placeholder response from a fallback system. Install with 'pip install llama-cpp-python' to use GGUF models.",
                        "ERROR: Cannot load GGUF model without llama-cpp-python. This is a placeholder response from a fallback system. Please check the installation instructions in the documentation.",
                        "ERROR: GGUF model requires llama-cpp-python. This is a placeholder response from a fallback system. Install the package and try again."
                    ],
                    sequential=True
                )
    else:
        logger.info(f"Using LLM model from HuggingFace Hub: {model_name}")
        model_location = model_name
    
    # Safely import torch and transformers
    torch = safe_import_torch()
    if torch is None:
        logger.error("Failed to safely import PyTorch. Using a mock LLM instead.")
        from langchain_community.llms.fake import FakeListLLM
        return FakeListLLM(
            responses=[
                "ERROR: Failed to safely import PyTorch. This is a placeholder response from a fallback system. Please check your PyTorch installation and ensure it's compatible with your environment.",
                "ERROR: PyTorch import failed. This is a placeholder response from a fallback system. Try reinstalling PyTorch with 'conda install -c pytorch pytorch' for better compatibility.",
                "ERROR: PyTorch could not be loaded. This is a placeholder response from a fallback system. Please check the logs for detailed error information.",
                "ERROR: PyTorch initialization failed. This is a placeholder response from a fallback system. Make sure your environment has the correct dependencies installed."
            ],
            sequential=True
        )
    
    transformers_modules = safe_import_transformers()
    if transformers_modules is None:
        logger.error("Failed to safely import transformers. Using a mock LLM instead.")
        from langchain_community.llms.fake import FakeListLLM
        return FakeListLLM(
            responses=[
                "ERROR: Failed to safely import transformers. This is a placeholder response from a fallback system. Please check your transformers installation and ensure it's compatible with your environment.",
                "ERROR: Transformers import failed. This is a placeholder response from a fallback system. Try reinstalling transformers with 'pip install transformers --upgrade' for better compatibility.",
                "ERROR: Transformers library could not be loaded. This is a placeholder response from a fallback system. Please check the logs for detailed error information.",
                "ERROR: Transformers initialization failed. This is a placeholder response from a fallback system. Make sure your environment has the correct dependencies installed."
            ],
            sequential=True
        )
    
    # Extract transformers modules
    AutoTokenizer = transformers_modules["AutoTokenizer"]
    AutoModelForCausalLM = transformers_modules["AutoModelForCausalLM"]
    pipeline = transformers_modules["pipeline"]
    BitsAndBytesConfig = transformers_modules["BitsAndBytesConfig"]
    
    # Configure quantization if GPU is available
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using quantized model.")
        if quantization_level == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization_level == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None
    else:
        logger.info("CUDA is not available. Using CPU.")
        quantization_config = None
    
    try:
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
        from langchain_community.llms import HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Fall back to a mock LLM that provides more useful responses
        from langchain_community.llms.fake import FakeListLLM
        
        # Log the error for debugging
        logger.error(f"Detailed error loading model: {str(e)}")
        
        # Create a more informative mock LLM
        return FakeListLLM(
            responses=[
                f"ERROR: Unable to load LLM model due to: {str(e)}. Please check your model installation and compatibility. This is a placeholder response from a fallback system.",
                f"ERROR: LLM model failed to load. Error details: {str(e)}. This is a placeholder response from a fallback system.",
                f"ERROR: Model initialization failed with error: {str(e)}. This is a placeholder response from a fallback system.",
                f"ERROR: Model loading error: {str(e)}. This is a placeholder response from a fallback system."
            ],
            sequential=True
        )

def get_safe_embedding_model(model_name: Optional[str] = None):
    """
    Get an embedding model with Streamlit compatibility.
    
    Args:
        model_name: Name of the embedding model to use. If None, uses the default model.
        
    Returns:
        An initialized embedding model instance.
    """
    from backend.config import (
        MODELS_DIR,
        DEFAULT_EMBEDDING_MODEL
    )
    
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
        # Safely import torch
        torch = safe_import_torch()
        if torch is None:
            logger.warning("Failed to safely import PyTorch. Using FakeEmbeddings instead.")
            logger.error("PyTorch is required for proper embedding model functionality")
            
            # Create a more informative error message for the logs
            error_message = """
ERROR: Failed to safely import PyTorch for embedding model. Using fake embeddings as fallback.
This will allow the application to run, but search results will not be accurate.
To fix this issue:
1. Check your PyTorch installation
2. Try reinstalling PyTorch with 'conda install -c pytorch pytorch' for better compatibility
3. Check the logs for detailed error information
"""
            logger.error(error_message)
            
            from langchain_community.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=768)
        
        # Check if model exists locally
        model_path = os.path.join(MODELS_DIR, os.path.basename(model_name))
        if os.path.exists(model_path):
            logger.info(f"Using local embedding model from: {model_path}")
            model_location = model_path
        else:
            logger.info(f"Using embedding model from HuggingFace Hub: {model_name}")
            model_location = model_name
        
        # Initialize and return the embedding model
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_location,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        logger.warning(f"Error loading embedding model: {e}")
        logger.warning("Using FakeEmbeddings as a fallback")
        logger.warning("This will allow the application to run, but search results may not be meaningful")
        logger.error(f"Detailed error loading embedding model: {str(e)}")
        
        # Create a more informative error message for the logs
        error_message = f"""
ERROR: Failed to load embedding model. Using fake embeddings as fallback.
Error details: {str(e)}
This will allow the application to run, but search results will not be accurate.
To fix this issue:
1. Check your sentence-transformers installation
2. Try reinstalling with 'pip install sentence-transformers --upgrade'
3. Ensure PyTorch is properly installed
4. Check the logs for more detailed error information
"""
        logger.error(error_message)
        
        from langchain_community.embeddings import FakeEmbeddings
        return FakeEmbeddings(size=768)
