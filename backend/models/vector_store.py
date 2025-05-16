"""
Vector database utilities for document storage and retrieval.
"""
import os
import logging
from typing import List, Dict, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import (
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    NUM_RETRIEVAL_RESULTS
)
from backend.config import EMBEDDING_TYPE
from backend.models.streamlit_compatibility import get_safe_embedding_model

logger = logging.getLogger(__name__)

def get_embedding_model() -> Embeddings:
    """
    Get the embedding model based on configuration.
    
    Returns:
        An initialized embedding model.
    """
    if EMBEDDING_TYPE == "llama_cpp":
        # Use llama.cpp embeddings
        try:
            from backend.models.embeddings_local import embeddings
            logger.info("Using llama.cpp embeddings")
            return embeddings
        except ImportError as e:
            logger.error(f"Error importing llama.cpp embeddings: {e}")
            logger.error("Falling back to safe embedding model")
            return get_safe_embedding_model()
    elif EMBEDDING_TYPE == "tensorflow":
        # Use TensorFlow embeddings
        try:
            from backend.models.alternative_embeddings import create_alternative_embeddings
            logger.info("Using TensorFlow embeddings")
            return create_alternative_embeddings("tensorflow")
        except ImportError as e:
            logger.error(f"Error importing TensorFlow embeddings: {e}")
            logger.error("Falling back to safe embedding model")
            return get_safe_embedding_model()
    elif EMBEDDING_TYPE == "tfidf":
        # Use TF-IDF embeddings
        try:
            from backend.models.alternative_embeddings import create_alternative_embeddings
            logger.info("Using TF-IDF embeddings")
            return create_alternative_embeddings("tfidf")
        except ImportError as e:
            logger.error(f"Error importing TF-IDF embeddings: {e}")
            logger.error("Falling back to safe embedding model")
            return get_safe_embedding_model()
    else:
        # Default to sentence-transformers embeddings
        logger.info("Using sentence-transformers embeddings")
        return get_safe_embedding_model()

def get_vector_store(
    embedding_model: Optional[Embeddings] = None,
    collection_name: str = "default"
) -> Chroma:
    """
    Initialize and return a ChromaDB vector store.
    
    Args:
        embedding_model: The embedding model to use. If None, uses the default model.
        collection_name: The name of the collection to use.
    
    Returns:
        An initialized Chroma vector store.
    """
    embedding_model = embedding_model or get_safe_embedding_model()
    
    # Create the vector store directory if it doesn't exist
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    
    # Initialize and return the vector store
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=VECTOR_DB_DIR
    )

def add_documents_to_vector_store(
    documents: List[Document],
    vector_store: Optional[Chroma] = None,
    embedding_model: Optional[Embeddings] = None,
    collection_name: str = "default",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> Chroma:
    """
    Add documents to the vector store.
    
    Args:
        documents: List of documents to add.
        vector_store: The vector store to add documents to. If None, creates a new one.
        embedding_model: The embedding model to use. If None, uses the default model.
        collection_name: The name of the collection to use.
        chunk_size: The size of text chunks to split documents into.
        chunk_overlap: The overlap between text chunks.
    
    Returns:
        The updated vector store.
    """
    # Initialize vector store if not provided
    if vector_store is None:
        vector_store = get_vector_store(
            embedding_model=embedding_model,
            collection_name=collection_name
        )
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_documents = text_splitter.split_documents(documents)
    
    logger.info(f"Adding {len(split_documents)} document chunks to vector store")
    
    # Add documents to vector store
    vector_store.add_documents(split_documents)
    
    return vector_store

def get_relevant_documents(
    query: str,
    vector_store: Optional[Chroma] = None,
    embedding_model: Optional[Embeddings] = None,
    collection_name: str = "default",
    num_results: int = NUM_RETRIEVAL_RESULTS
) -> List[Document]:
    """
    Retrieve relevant documents from the vector store based on a query.
    
    Args:
        query: The query to search for.
        vector_store: The vector store to search in. If None, creates a new one.
        embedding_model: The embedding model to use. If None, uses the default model.
        collection_name: The name of the collection to use.
        num_results: The number of results to return.
    
    Returns:
        A list of relevant documents.
    """
    # Initialize vector store if not provided
    if vector_store is None:
        vector_store = get_vector_store(
            embedding_model=embedding_model,
            collection_name=collection_name
        )
    
    # Retrieve and return relevant documents
    return vector_store.similarity_search(query, k=num_results)

def get_retriever(
    vector_store: Optional[Chroma] = None,
    embedding_model: Optional[Embeddings] = None,
    collection_name: str = "default",
    search_kwargs: Optional[Dict] = None
):
    """
    Get a retriever from the vector store.
    
    Args:
        vector_store: The vector store to get a retriever from. If None, creates a new one.
        embedding_model: The embedding model to use. If None, uses the default model.
        collection_name: The name of the collection to use.
        search_kwargs: Additional search parameters.
    
    Returns:
        A retriever from the vector store.
    """
    # Initialize vector store if not provided
    if vector_store is None:
        vector_store = get_vector_store(
            embedding_model=embedding_model,
            collection_name=collection_name
        )
    
    # Set default search kwargs if not provided
    if search_kwargs is None:
        search_kwargs = {"k": 10}
    
    # Return the retriever
    return vector_store.as_retriever(search_kwargs=search_kwargs)
