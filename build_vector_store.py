"""
Script to build the vector store for Database Copilot.
This is separate from setup.py to allow users to add custom documents
before creating the vector store.
"""
import argparse
import logging
import sys
from backend.data_ingestion.ingest import ingest_all_documents, ingest_documents
from backend.config import DOC_CATEGORIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the vector store building script.
    """
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
        logger.info("You can now run the application with: python run_app.py")
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
