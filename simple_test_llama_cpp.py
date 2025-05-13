#!/usr/bin/env python3
"""
Simple test script for llama-cpp-python.
"""
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Test llama-cpp-python.
    """
    try:
        # Print NumPy version
        logger.info(f"NumPy version: {np.__version__}")
        
        # Import llama_cpp
        import llama_cpp
        logger.info(f"llama_cpp version: {llama_cpp.__version__}")
        
        # Set up the model path
        model_path = os.path.join("data", "hf_models", "nomic-embed-text-v1.5.Q4_K_M.gguf")
        
        # Check if the model exists
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return 1
        
        # Initialize the model
        logger.info(f"Initializing llama.cpp model: {model_path}")
        model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=4,
            embedding=True
        )
        
        # Test the embeddings
        test_text = "This is a test sentence for embeddings."
        
        logger.info("Generating embeddings for test text...")
        embedding = model.embed(test_text)
        
        # Print the results
        logger.info(f"Embedding dimension: {len(embedding)}")
        logger.info("Embeddings generated successfully!")
        
        return 0
    except Exception as e:
        logger.error(f"Error testing llama-cpp-python: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
