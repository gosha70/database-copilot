"""
Local embeddings implementation using llama.cpp.

This module provides embeddings using llama.cpp, which works well on Apple Silicon
and doesn't require PyTorch or sentence-transformers.
"""
import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class LlamaCppEmbeddings(Embeddings):
    """
    Llama.cpp-based embeddings using nomic-embed-text model.
    
    This class provides embeddings using llama.cpp, which works well on
    Apple Silicon and doesn't require PyTorch or sentence-transformers.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 8192,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = -1,
        verbose: bool = False
    ):
        """
        Initialize the llama.cpp embeddings.
        
        Args:
            model_path: Path to the GGUF model file. If None, uses the default path.
            n_ctx: Maximum context size.
            n_threads: Number of threads to use. If None, uses all available cores.
            n_gpu_layers: Number of layers to offload to GPU. -1 means all layers.
            verbose: Whether to enable verbose output.
        """
        try:
            from langchain_community.embeddings import LlamaCppEmbeddings as LCLlamaCppEmbeddings
            
            # Set default model path if not provided
            if model_path is None:
                # Get the default model path from config
                from backend.config import MODELS_DIR, DEFAULT_LLAMA_CPP_EMBEDDING_MODEL
                model_path = os.path.join(MODELS_DIR, DEFAULT_LLAMA_CPP_EMBEDDING_MODEL)
            
            # Set default number of threads if not provided
            if n_threads is None:
                import multiprocessing
                n_threads = max(1, multiprocessing.cpu_count() // 2)
            
            logger.info(f"Initializing llama.cpp embeddings with model: {model_path}")
            logger.info(f"Using {n_threads} threads and {n_gpu_layers} GPU layers")
            
            # Initialize the embeddings
            self.embeddings = LCLlamaCppEmbeddings(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
            
            logger.info("Llama.cpp embeddings initialized successfully")
            
        except ImportError as e:
            logger.error(f"Error importing llama_cpp: {e}")
            logger.error("Please install llama-cpp-python with:")
            logger.error("pip install llama-cpp-python")
            raise ImportError(f"Failed to import llama_cpp: {e}")
        except Exception as e:
            logger.error(f"Error initializing llama.cpp embeddings: {e}")
            raise ValueError(f"Failed to initialize llama.cpp embeddings: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: A list of documents to embed.
            
        Returns:
            A list of embeddings, one for each document.
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return zero embeddings as fallback
            # Use a reasonable embedding dimension (384 for nomic-embed-text)
            return [[0.0] * 384 for _ in range(len(texts))]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query.
        
        Args:
            text: The query to embed.
            
        Returns:
            The embedding for the query.
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            # Return zero embedding as fallback
            # Use a reasonable embedding dimension (384 for nomic-embed-text)
            return [0.0] * 384


# Create a singleton instance of the embeddings
_embeddings_instance = None

def get_embeddings() -> Embeddings:
    """
    Get a singleton instance of the embeddings.
    
    Returns:
        An initialized embeddings instance.
    """
    global _embeddings_instance
    
    if _embeddings_instance is None:
        _embeddings_instance = LlamaCppEmbeddings()
    
    return _embeddings_instance


# Expose the embeddings instance directly
embeddings = get_embeddings()
