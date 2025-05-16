#!/usr/bin/env python3
"""
Script to rebuild the vector store using llama.cpp embeddings.

This script rebuilds all collections in the vector store using the
nomic-embed-text-v1.5.Q4_K_M.gguf model via llama.cpp.
"""
import os
import logging
import argparse
import sys
from pathlib import Path

# Check for required dependencies
missing_deps = []
try:
    import langchain_core
except ImportError:
    missing_deps.append("langchain-core>=0.1.10")

try:
    import langchain_community
except ImportError:
    missing_deps.append("langchain-community>=0.0.10")

try:
    import chromadb
except ImportError:
    missing_deps.append("chromadb>=0.4.18")

# If dependencies are missing, print instructions and exit
if missing_deps:
    print("Error: Missing required dependencies:")
    for dep in missing_deps:
        print(f"  - {dep}")
    print("\nPlease install the missing dependencies with one of these commands:")
    print(f"  pip install {' '.join(missing_deps)}")
    print(f"  pip3 install {' '.join(missing_deps)}")
    print(f"  python -m pip install {' '.join(missing_deps)}")
    print(f"  python3 -m pip install {' '.join(missing_deps)}")
    print("\nIf you're using a virtual environment, make sure it's activated first.")
    print("If you're using conda, make sure the conda environment is activated first.")
    print("\nAlternatively, you can run the setup script which will install all dependencies:")
    print("  ./run_torch_free.sh setup")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ensure_model_exists():
    """
    Ensure that the embedding model exists.
    
    Returns:
        bool: True if the model exists or was successfully downloaded, False otherwise.
    """
    # Import config to get the model path
    from backend.config import MODELS_DIR, DEFAULT_LLAMA_CPP_EMBEDDING_MODEL
    
    # Check if the model exists
    model_path = os.path.join(MODELS_DIR, DEFAULT_LLAMA_CPP_EMBEDDING_MODEL)
    if os.path.exists(model_path):
        logger.info(f"Embedding model found at {model_path}")
        return True
    
    # If the model doesn't exist, download it
    logger.info(f"Embedding model not found at {model_path}")
    logger.info("Downloading the model...")
    
    try:
        # Import and run the download script
        import download_nomic_embed
        result = download_nomic_embed.main()
        
        if result == 0:
            logger.info("Model downloaded successfully")
            return True
        else:
            logger.error("Failed to download the model")
            return False
    except Exception as e:
        logger.error(f"Error downloading the model: {e}")
        return False

def main():
    """
    Main function to rebuild the vector store.
    """
    parser = argparse.ArgumentParser(description="Rebuild vector store with llama.cpp embeddings")
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Document category to rebuild (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure the embedding model exists
    if not ensure_model_exists():
        logger.error("Cannot proceed without the embedding model")
        return 1
    
    # Set environment variable to use llama.cpp embeddings
    os.environ["EMBEDDING_TYPE"] = "llama_cpp"
    logger.info("Using llama.cpp embeddings for vector store")
    
    try:
        # Import the ingest module
        from backend.config import DOC_CATEGORIES
        from backend.data_ingestion.ingest import ingest_all_documents, ingest_documents
        
        if args.category == "all":
            logger.info("Rebuilding all vector store collections")
            logger.info("This will process documents from the following directories:")
            for category, doc_dir in DOC_CATEGORIES.items():
                logger.info(f"  - {category}: {doc_dir}")
            
            ingest_all_documents(recreate=True)
        else:
            # Check if the category exists
            if args.category not in DOC_CATEGORIES:
                logger.error(f"Unknown category: {args.category}")
                logger.error(f"Available categories: {', '.join(DOC_CATEGORIES.keys())}")
                return 1
            
            doc_dir = DOC_CATEGORIES[args.category]
            logger.info(f"Rebuilding vector store for category: {args.category}")
            logger.info(f"Processing documents from: {doc_dir}")
            
            ingest_documents(
                doc_dir=doc_dir,
                collection_name=args.category,
                recreate=True
            )
        
        logger.info("Vector store rebuilt successfully with llama.cpp embeddings")
        logger.info("You can now run the application with: ./run_torch_free.sh run")
    except Exception as e:
        logger.error(f"Failed to rebuild vector store: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
