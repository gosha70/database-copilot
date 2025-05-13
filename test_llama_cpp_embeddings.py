#!/usr/bin/env python3
"""
Test script for llama.cpp embeddings.
"""
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Test llama.cpp embeddings.
    """
    # Add the current directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    try:
        # Import the embeddings
        from backend.models.embeddings_local import LlamaCppEmbeddings
        
        # Set up the model path
        model_path = os.path.join("data", "hf_models", "nomic-embed-text-v1.5.Q4_K_M.gguf")
        
        # Check if the model exists
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            logger.info("Downloading the model...")
            
            # Import and run the download script
            import download_nomic_embed
            result = download_nomic_embed.main()
            
            if result != 0:
                logger.error("Failed to download the model")
                return 1
        
        # Initialize the embeddings
        logger.info(f"Initializing llama.cpp embeddings with model: {model_path}")
        embeddings = LlamaCppEmbeddings(model_path=model_path)
        
        # Test the embeddings
        test_texts = [
            "This is a test sentence for embeddings.",
            "Another example to check if embeddings work."
        ]
        
        logger.info("Generating embeddings for test texts...")
        result = embeddings.embed_documents(test_texts)
        
        # Print the results
        logger.info(f"Generated {len(result)} embeddings")
        logger.info(f"Embedding dimension: {len(result[0])}")
        logger.info("Embeddings generated successfully!")
        
        return 0
    except Exception as e:
        logger.error(f"Error testing llama.cpp embeddings: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
