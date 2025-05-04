"""
Script to ingest documents into the vector database.
"""
import os
import logging
import argparse
from typing import List, Dict, Optional

from backend.config import DOC_CATEGORIES
from backend.data_ingestion.document_loaders import load_documents
from backend.models.vector_store import add_documents_to_vector_store, get_vector_store
from backend.models.llm import get_embedding_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ingest_documents(
    doc_dir: str,
    collection_name: str = "default",
    recreate: bool = False
) -> None:
    """
    Ingest documents from a directory into the vector database.
    
    Args:
        doc_dir: Path to the directory containing documents.
        collection_name: Name of the collection to store documents in.
        recreate: Whether to recreate the collection if it already exists.
    """
    logger.info(f"Ingesting documents from {doc_dir} into collection {collection_name}")
    
    # Load documents
    documents = load_documents(doc_dir)
    logger.info(f"Loaded {len(documents)} documents")
    
    if not documents:
        logger.warning(f"No documents found in {doc_dir}")
        return
    
    # Get embedding model
    embedding_model = get_embedding_model()
    
    # Get vector store
    vector_store = get_vector_store(
        embedding_model=embedding_model,
        collection_name=collection_name
    )
    
    # Check if collection should be recreated
    if recreate:
        logger.info(f"Recreating collection {collection_name}")
        vector_store.delete_collection()
        vector_store = get_vector_store(
            embedding_model=embedding_model,
            collection_name=collection_name
        )
    
    # Add documents to vector store
    add_documents_to_vector_store(
        documents=documents,
        vector_store=vector_store,
        embedding_model=embedding_model,
        collection_name=collection_name
    )
    
    logger.info(f"Successfully ingested documents into collection {collection_name}")

def ingest_all_documents(recreate: bool = False) -> None:
    """
    Ingest all documents from configured categories into the vector database.
    
    Args:
        recreate: Whether to recreate collections if they already exist.
    """
    logger.info("Ingesting all documents")
    
    for category, doc_dir in DOC_CATEGORIES.items():
        # Create directory if it doesn't exist
        os.makedirs(doc_dir, exist_ok=True)
        
        # Check if directory is empty
        if not os.listdir(doc_dir):
            logger.warning(f"Directory {doc_dir} is empty. Skipping ingestion for {category}.")
            continue
        
        # Ingest documents
        ingest_documents(
            doc_dir=doc_dir,
            collection_name=category,
            recreate=recreate
        )
    
    logger.info("Successfully ingested all documents")

def main():
    """
    Main function to run the ingestion script.
    """
    parser = argparse.ArgumentParser(description="Ingest documents into the vector database")
    parser.add_argument(
        "--category",
        type=str,
        choices=list(DOC_CATEGORIES.keys()) + ["all"],
        default="all",
        help="Document category to ingest"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collections if they already exist"
    )
    
    args = parser.parse_args()
    
    if args.category == "all":
        ingest_all_documents(recreate=args.recreate)
    else:
        doc_dir = DOC_CATEGORIES[args.category]
        ingest_documents(
            doc_dir=doc_dir,
            collection_name=args.category,
            recreate=args.recreate
        )

if __name__ == "__main__":
    main()
