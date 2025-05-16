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
        # Completely bypass the problematic torch._classes module
        # by creating a fake module structure before importing torch
        # Completely bypass the problematic torch._classes module
        # by creating a fake module structure before importing torch
        import types
        import sys
        
        # Create a complete fake module hierarchy
        fake_torch = types.ModuleType("torch")
        fake_c = types.ModuleType("_C")
        fake_classes = types.ModuleType("_classes")
        
        # Create a fake cuda module with proper methods
        class FakeCuda:
            def __init__(self):
                pass
            
            def is_available(self):
                return False
            
            def device_count(self):
                return 0
        
        # Add cuda to fake_torch
        fake_cuda = FakeCuda()
        fake_torch.cuda = fake_cuda
        # Also add is_available as a direct attribute for compatibility
        fake_torch.cuda.is_available = fake_cuda.is_available
        
        # Set up the fake path attribute that won't cause issues
        class FakePath:
            def __init__(self):
                self._path = ["/dummy/path"]
            
            def __iter__(self):
                return iter(self._path)
            
            def __getattr__(self, name):
                if name == "_path":
                    return self._path
                return None
        
        fake_path = FakePath()
        fake_classes.__path__ = fake_path
        
        # Set up a safe __getattr__ that won't try to access custom classes
        def safe_getattr(self, name):
            if name == "__path__":
                return fake_path
            return None
        
        # Apply the safe __getattr__ to the fake classes module
        fake_classes.__getattr__ = lambda attr: None
        fake_c.__getattr__ = lambda attr: None
        import sys
        
        # Create a complete fake module hierarchy
        fake_torch = types.ModuleType("torch")
        fake_c = types.ModuleType("_C")
        fake_classes = types.ModuleType("_classes")
        
        # Create a fake cuda module with proper methods
        class FakeCuda:
            def __init__(self):
                pass
            
            def is_available(self):
                return False
            
            def device_count(self):
                return 0
        
        # Add cuda to fake_torch
        fake_cuda = FakeCuda()
        fake_torch.cuda = fake_cuda
        # Also add is_available as a direct attribute for compatibility
        fake_torch.cuda.is_available = fake_cuda.is_available
        
        # Set up the fake path attribute that won't cause issues
        class FakePath:
            def __init__(self):
                self._path = ["/dummy/path"]
            
            def __iter__(self):
                return iter(self._path)
            
            def __getattr__(self, name):
                if name == "_path":
                    return self._path
                return None
        
        fake_path = FakePath()
        fake_classes.__path__ = fake_path
        
        # Set up a safe __getattr__ that won't try to access custom classes
        def safe_getattr(self, name):
            if name == "__path__":
                return fake_path
            return None
        
        # Apply the safe __getattr__ to the fake classes module
        fake_classes.__getattr__ = lambda attr: None
        fake_c.__getattr__ = lambda attr: None
        
        # Set up the module hierarchy
        fake_torch._C = fake_c
        fake_torch._classes = fake_classes
        
        # Register the fake modules in sys.modules
        sys.modules["torch"] = fake_torch
        sys.modules["torch._C"] = fake_c
        sys.modules["torch._classes"] = fake_classes
        # Set up the module hierarchy
        fake_torch._C = fake_c
        fake_torch._classes = fake_classes
        
        # Register the fake modules in sys.modules
        sys.modules["torch"] = fake_torch
        sys.modules["torch._C"] = fake_c
        sys.modules["torch._classes"] = fake_classes
        
        # Now import the real torch, which will replace our fake modules
        # but keep our safe __getattr__ methods
        # Now import the real torch, which will replace our fake modules
        # but keep our safe __getattr__ methods
        import torch
        
        # Ensure our safe __getattr__ is still used for _classes
        
        # Ensure our safe __getattr__ is still used for _classes
        if hasattr(torch, "_classes"):
            torch._classes.__getattr__ = lambda attr: None
            torch._classes.__getattr__ = lambda attr: None
        
        # Mark as successfully imported
        _TORCH_IMPORTED = True
        
        return torch
    
    except Exception as e:
        logger.error(f"Error safely importing PyTorch: {e}")
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

def get_safe_llm(model_name: Optional[str] = None, quantization_level: str = "4bit", use_external: bool = False):
    """
    Get an LLM with Streamlit compatibility.
    
    Args:
        model_name: Name of the LLM model to use. If None, uses the default model.
        quantization_level: The quantization level to use (4bit, 8bit, or none)
        use_external: Whether to use an external LLM if configured
        
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
        REPETITION_PENALTY,
        get_current_llm_type
    )
    
    # Get the current LLM type
    current_llm_type = get_current_llm_type()
    
    # Check if we should use an external LLM
    if use_external and current_llm_type != "local":
        try:
            from backend.models.external_llm.factory import create_external_llm
            logger.info(f"Creating external LLM of type: {current_llm_type}")
            external_llm = create_external_llm(current_llm_type)
            if external_llm:
                return external_llm
            else:
                logger.error(f"Failed to create external LLM of type {current_llm_type}")
                raise ValueError(f"Failed to create external LLM of type {current_llm_type}. Check your API key and configuration.")
        except Exception as e:
            logger.error(f"Error initializing external LLM: {e}")
            raise ValueError(f"Error initializing external LLM: {e}. Check your API key and configuration.")
    
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
                error_message = (
                    "GGUF model detected but llama-cpp-python is not installed. "
                    "Please install it with: pip install llama-cpp-python"
                )
                logger.error(error_message)
                raise ImportError(error_message)
    else:
        logger.info(f"Using LLM model from HuggingFace Hub: {model_name}")
        model_location = model_name
    
    # Safely import torch and transformers
    torch = safe_import_torch()
    if torch is None:
        error_message = "Failed to safely import PyTorch. Please check your PyTorch installation."
        logger.error(error_message)
        logger.error("For Apple M1/M2 Mac users:")
        logger.error("1. Install PyTorch with MPS support: pip install torch torchvision torchaudio")
        logger.error("2. Make sure you're using Python 3.9+ for best compatibility")
        logger.error("3. If using Conda: conda install pytorch torchvision torchaudio -c pytorch-nightly")
        raise ImportError(error_message)
    
    transformers_modules = safe_import_transformers()
    if transformers_modules is None:
        error_message = "Failed to safely import transformers. Please check your transformers installation."
        logger.error(error_message)
        logger.error("Try reinstalling with: pip install transformers --upgrade")
        raise ImportError(error_message)
    
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
        error_message = f"Error loading model: {e}"
        logger.error(error_message)
        logger.error(f"Detailed error loading model: {str(e)}")
        logger.error("Please check your model installation and compatibility.")
        raise RuntimeError(error_message)

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
        DEFAULT_EMBEDDING_MODEL,
        EMBEDDING_TYPE
    )
    
    # Check if we should use llama.cpp embeddings
    if EMBEDDING_TYPE == "llama_cpp":
        try:
            from backend.models.embeddings_local import embeddings
            logger.info("Using llama.cpp embeddings")
            return embeddings
        except ImportError as e:
            logger.error(f"Error importing llama.cpp embeddings: {e}")
            logger.error("Make sure llama-cpp-python is installed")
            # Continue to try sentence_transformers as fallback
    
    # If we're here, we're using sentence_transformers
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
            logger.error("Failed to safely import PyTorch. Cannot proceed with sentence_transformers embedding model.")
            
            # Create a more informative error message for the logs
            error_message = """
ERROR: Failed to safely import PyTorch for embedding model.
For Apple M1/M2 Mac users:
1. Install PyTorch with MPS support: pip install torch torchvision torchaudio
2. Make sure you're using Python 3.9+ for best compatibility
3. If using Conda: conda install pytorch torchvision torchaudio -c pytorch-nightly
4. Alternatively, use the torch-free setup: ./run_torch_free.sh setup

For other users:
1. Check your PyTorch installation
2. Try reinstalling PyTorch with the appropriate command for your system
   - For CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   - For CPU only: pip install torch torchvision torchaudio
"""
            logger.error(error_message)
            
            # Raise an exception to stop the application
            raise ImportError("PyTorch failed to load. See logs for installation instructions.")
        
        # Check if model exists locally
        model_path = os.path.join(MODELS_DIR, os.path.basename(model_name))
        if os.path.exists(model_path):
            logger.info(f"Using local embedding model from: {model_path}")
            model_location = model_path
        else:
            logger.info(f"Using embedding model from HuggingFace Hub: {model_name}")
            model_location = model_name
        
        # Try to directly import sentence_transformers to diagnose the issue
        try:
            import sentence_transformers
            logger.info(f"Successfully imported sentence_transformers from {sentence_transformers.__file__}")
        except ImportError as e:
            logger.error(f"Direct import of sentence_transformers failed: {e}")
            
            # Try to use llama.cpp embeddings as fallback
            try:
                from backend.models.embeddings_local import embeddings
                logger.info("Falling back to llama.cpp embeddings due to sentence_transformers import failure")
                return embeddings
            except ImportError as sub_e:
                logger.error(f"Error importing llama.cpp embeddings: {sub_e}")
            
            # Try to find the package in the Python path
            import subprocess
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "show", "sentence-transformers"], 
                                        capture_output=True, text=True)
                logger.info(f"Pip show sentence-transformers result:\n{result.stdout}")
            except Exception as sub_e:
                logger.error(f"Failed to run pip show: {sub_e}")
            
            # Raise the error to stop execution
            raise ImportError(f"Failed to import sentence_transformers directly: {e}. This is likely a Python environment issue.")
        
        # Initialize and return the embedding model
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_location,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        error_str = str(e)
        logger.error(f"Error loading embedding model: {error_str}")
        
        # Check for specific error types and provide targeted solutions
        if "Could not import sentence_transformers" in error_str:
            # This is a Python path/environment issue
            import sys
            import site

            # Log detailed environment information for debugging
            logger.error(f"Python executable: {sys.executable}")
            logger.error(f"Python version: {sys.version}")
            logger.error(f"Python path: {sys.path}")
            logger.error(f"Site packages: {site.getsitepackages()}")

            # Create a more specific error message for sentence_transformers import issues
            error_message = f"""
ERROR: Failed to import sentence_transformers package.
This is likely a Python environment issue, not an installation issue.

Debugging information:
- Python executable: {sys.executable}
- Python version: {sys.version.split()[0]}

For Apple M1/M2 Mac users:
1. Make sure you're running the app with the SAME Python environment where you installed sentence-transformers
2. Try installing directly in your current environment:
   {sys.executable} -m pip install --force-reinstall sentence-transformers

3. If using a virtual environment, activate it before running the app:
   source /path/to/your/venv/bin/activate  # Replace with your actual path

4. If using Conda, make sure you've activated the correct environment:
   conda activate your_environment_name

5. Check if the package is installed but in a different location:
   {sys.executable} -m pip show sentence-transformers
"""
            logger.error(error_message)

            # Raise an exception with the specific guidance
            raise ImportError(f"Failed to import sentence_transformers. This is a Python environment issue, not an installation issue. Run the app with the same Python environment where you installed the package. See logs for detailed debugging information.")
        else:
            # Generic error message for other types of errors
            error_message = f"""
ERROR: Failed to load embedding model.
Error details: {error_str}

For Apple M1/M2 Mac users:
1. Install PyTorch with MPS support: pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
2. Make sure you're using Python 3.9+ for best compatibility
3. If using Conda: conda install pytorch torchvision torchaudio -c pytorch

For other users:
1. Check your sentence-transformers installation
2. Try reinstalling with 'pip install sentence-transformers --upgrade'
3. Ensure PyTorch is properly installed
4. Check the logs for more detailed error information
"""
            logger.error(error_message)

            # Raise an exception to stop the application
            raise ImportError(f"Failed to load embedding model: {error_str}. See logs for installation instructions.")
