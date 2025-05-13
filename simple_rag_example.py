#!/usr/bin/env python3
"""
Simple RAG example using llama.cpp for both embeddings and LLM.

This script demonstrates a PyTorch-free RAG system using llama.cpp
for both embeddings and LLM generation.
"""
import os
import logging
import argparse
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_models():
    """
    Set up the embedding model and LLM using llama.cpp.
    
    Returns:
        tuple: (embedding_model, llm_model)
    """
    try:
        # Import llama_cpp
        import llama_cpp
        
        # Set up the embedding model path
        embedding_model_path = os.path.join("data", "hf_models", "nomic-embed-text-v1.5.Q4_K_M.gguf")
        
        # Check if the embedding model exists
        if not os.path.exists(embedding_model_path):
            logger.error(f"Embedding model not found at {embedding_model_path}")
            logger.info("Please run download_nomic_embed.py first")
            return None, None
        
        # Set up the LLM model path
        llm_model_path = os.path.join("data", "hf_models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        
        # Check if the LLM model exists
        if not os.path.exists(llm_model_path):
            logger.error(f"LLM model not found at {llm_model_path}")
            logger.info("Please download the LLM model first")
            return None, None
        
        # Initialize the embedding model
        logger.info(f"Initializing embedding model: {embedding_model_path}")
        embedding_model = llama_cpp.Llama(
            model_path=embedding_model_path,
            n_ctx=512,
            n_threads=4,
            embedding=True
        )
        
        # Initialize the LLM model
        logger.info(f"Initializing LLM model: {llm_model_path}")
        llm_model = llama_cpp.Llama(
            model_path=llm_model_path,
            n_ctx=4096,
            n_threads=4,
            n_gpu_layers=-1  # Use all GPU layers
        )
        
        return embedding_model, llm_model
    
    except Exception as e:
        logger.error(f"Error setting up models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def create_vector_store(embedding_model, documents: List[Dict[str, str]]):
    """
    Create a simple in-memory vector store.
    
    Args:
        embedding_model: The embedding model to use
        documents: List of documents to add to the vector store
    
    Returns:
        list: List of document embeddings with metadata
    """
    vector_store = []
    
    for doc in documents:
        # Generate embedding for the document
        embedding = embedding_model.embed(doc["content"])
        
        # Add the document and its embedding to the vector store
        vector_store.append({
            "embedding": embedding,
            "metadata": doc
        })
    
    return vector_store

def search_vector_store(embedding_model, vector_store, query: str, top_k: int = 3):
    """
    Search the vector store for documents similar to the query.
    
    Args:
        embedding_model: The embedding model to use
        vector_store: The vector store to search
        query: The query to search for
        top_k: The number of results to return
    
    Returns:
        list: List of top_k most similar documents
    """
    import numpy as np
    
    # Generate embedding for the query
    query_embedding = embedding_model.embed(query)
    
    # Calculate cosine similarity between query and all documents
    similarities = []
    for item in vector_store:
        doc_embedding = item["embedding"]
        
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        similarities.append((similarity, item["metadata"]))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Return top_k results
    return similarities[:top_k]

def generate_response(llm_model, query: str, context: str):
    """
    Generate a response using the LLM.
    
    Args:
        llm_model: The LLM model to use
        query: The user query
        context: The context from retrieved documents
    
    Returns:
        str: The generated response
    """
    # Create a prompt with the context and query
    prompt = f"""<s>[INST] You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {query} [/INST]
"""
    
    # Generate a response
    response = llm_model(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        echo=False
    )
    
    # Extract the generated text
    return response["choices"][0]["text"].strip()

def main():
    """
    Main function to run the RAG example.
    """
    # Set up the models
    embedding_model, llm_model = setup_models()
    if embedding_model is None or llm_model is None:
        return 1
    
    # Sample documents
    documents = [
        {
            "id": "doc1",
            "content": "Liquibase is an open-source database schema change management solution that enables you to manage revisions of your database schema scripts.",
            "title": "Introduction to Liquibase"
        },
        {
            "id": "doc2",
            "content": "Liquibase uses changesets to represent a single change to your database. Each changeset has an id and author attribute which, along with the changelog file path, uniquely identify it.",
            "title": "Liquibase Changesets"
        },
        {
            "id": "doc3",
            "content": "JPA (Java Persistence API) is a Java specification for accessing, persisting, and managing data between Java objects and a relational database.",
            "title": "Introduction to JPA"
        },
        {
            "id": "doc4",
            "content": "Hibernate is an Object-Relational Mapping (ORM) tool that implements the JPA specification and provides a framework for mapping Java objects to database tables.",
            "title": "Hibernate ORM"
        },
        {
            "id": "doc5",
            "content": "Entity classes in JPA are used to represent tables in a relational database. Each instance of an entity represents a row in the table.",
            "title": "JPA Entities"
        }
    ]
    
    # Create a vector store
    logger.info("Creating vector store...")
    vector_store = create_vector_store(embedding_model, documents)
    logger.info(f"Created vector store with {len(vector_store)} documents")
    
    # Example queries
    queries = [
        "What is Liquibase?",
        "How do JPA entities work?",
        "What is the relationship between Hibernate and JPA?"
    ]
    
    # Process each query
    for query in queries:
        logger.info(f"\nQuery: {query}")
        
        # Search for relevant documents
        results = search_vector_store(embedding_model, vector_store, query)
        
        # Create context from retrieved documents
        context = "\n\n".join([f"Document: {doc['title']}\n{doc['content']}" for _, doc in results])
        logger.info(f"Retrieved {len(results)} documents")
        
        # Generate a response
        logger.info("Generating response...")
        response = generate_response(llm_model, query, context)
        
        # Print the response
        logger.info(f"Response: {response}")
    
    return 0

if __name__ == "__main__":
    exit(main())
