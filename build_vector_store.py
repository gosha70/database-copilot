"""
Script to build the vector store for Database Copilot.
This is separate from setup.py to allow users to add custom documents
before creating the vector store.
"""
import argparse
import logging
import sys
import os
from pathlib import Path
from backend.data_ingestion.ingest import ingest_all_documents, ingest_documents
from backend.config import DOC_CATEGORIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ensure_secrets_file():
    """
    Ensure that a secrets.toml file exists.
    """
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    secrets_file = streamlit_dir / "secrets.toml"
    if not secrets_file.exists():
        with open(secrets_file, "w") as f:
            f.write("# Streamlit secrets file\n")
            f.write("# Uncomment and set values as needed\n\n")
            f.write("# LLM_TYPE = \"local\"\n\n")
            f.write("# OpenAI Configuration\n")
            f.write("# OPENAI_API_KEY = \"your-openai-api-key\"\n")
            f.write("# OPENAI_MODEL = \"gpt-4o\"\n\n")
            f.write("# Claude Configuration\n")
            f.write("# ANTHROPIC_API_KEY = \"your-anthropic-api-key\"\n")
            f.write("# CLAUDE_MODEL = \"claude-3-opus-20240229\"\n\n")
            f.write("# Gemini Configuration\n")
            f.write("# GOOGLE_API_KEY = \"your-google-api-key\"\n")
            f.write("# GEMINI_MODEL = \"gemini-1.5-pro\"\n\n")
            f.write("# Mistral Configuration\n")
            f.write("# MISTRAL_API_KEY = \"your-mistral-api-key\"\n")
            f.write("# MISTRAL_MODEL = \"mistral-medium\"\n\n")
            f.write("# DeepSeek Configuration\n")
            f.write("# DEEPSEEK_API_KEY = \"your-deepseek-api-key\"\n")
            f.write("# DEEPSEEK_MODEL = \"deepseek-chat\"\n")
        logger.info(f"Created empty secrets file at {secrets_file}")

def main():
    """
    Main function to run the vector store building script.
    """
    # Ensure secrets file exists
    ensure_secrets_file()
    
    parser = argparse.ArgumentParser(description="Build vector store for Database Copilot")
    parser.add_argument(
        "--category",
        type=str,
        choices=list(DOC_CATEGORIES.keys()) + ["all"],
        default="all",
        help="Document category to ingest (default: all)"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collections if they already exist"
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
    
    try:
        if args.category == "all":
            logger.info("Building vector store for all document categories")
            logger.info("This will process documents from the following directories:")
            for category, doc_dir in DOC_CATEGORIES.items():
                logger.info(f"  - {category}: {doc_dir}")
            
            ingest_all_documents(recreate=args.recreate)
        else:
            doc_dir = DOC_CATEGORIES[args.category]
            logger.info(f"Building vector store for category: {args.category}")
            logger.info(f"Processing documents from: {doc_dir}")
            
            ingest_documents(
                doc_dir=doc_dir,
                collection_name=args.category,
                recreate=args.recreate
            )
        
        logger.info("Vector store built successfully")
        logger.info("You can now run the application with: python run_app.py (this launches the optimized app)")
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
