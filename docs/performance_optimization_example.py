"""
Example implementation of performance optimizations for the Database Copilot.

This file demonstrates how to implement the performance optimizations
described in the enhancement plan, including caching, asynchronous processing,
and model optimization techniques.
"""
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
from typing import Dict, List, Optional, Union, Any, Callable
from functools import lru_cache
from collections import OrderedDict

import torch
from transformers import BitsAndBytesConfig
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

# ============================================================================
# Caching Implementations
# ============================================================================

class CachedEmbeddings:
    """
    A wrapper around an embedding model that caches results to avoid redundant computation.
    """
    
    def __init__(self, embedding_model: Embeddings, cache_size: int = 1000):
        """
        Initialize the cached embeddings.
        
        Args:
            embedding_model: The underlying embedding model
            cache_size: Maximum number of embeddings to cache
        """
        self.embedding_model = embedding_model
        self.cache_size = cache_size
        self.cache = OrderedDict()
    
    def _hash_text(self, text: str) -> str:
        """
        Create a hash of the text for use as a cache key.
        
        Args:
            text: The text to hash
            
        Returns:
            A hash of the text
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents, using cached values when available.
        
        Args:
            texts: The texts to embed
            
        Returns:
            A list of embeddings
        """
        # Check cache first
        cache_keys = [self._hash_text(text) for text in texts]
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        for i, (text, key) in enumerate(zip(texts, cache_keys)):
            if key in self.cache:
                # Move the item to the end of the OrderedDict to mark it as recently used
                self.cache.move_to_end(key)
                results[i] = self.cache[key]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed only uncached texts
        if texts_to_embed:
            logger.info(f"Embedding {len(texts_to_embed)} new texts (cache miss)")
            new_embeddings = self.embedding_model.embed_documents(texts_to_embed)
            
            # Update cache and results
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                key = cache_keys[idx]
                
                # If cache is full, remove the oldest item
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)
                
                self.cache[key] = embedding
                results[idx] = embedding
        else:
            logger.info(f"All {len(texts)} texts found in cache (cache hit)")
        
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text, using cached values when available.
        
        Args:
            text: The text to embed
            
        Returns:
            An embedding
        """
        # Use the same caching logic as embed_documents
        key = self._hash_text(text)
        
        if key in self.cache:
            # Move the item to the end of the OrderedDict to mark it as recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        
        # Embed the query
        embedding = self.embedding_model.embed_query(text)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = embedding
        
        return embedding


class CachedLLM:
    """
    A wrapper around an LLM that caches responses to avoid redundant computation.
    """
    
    def __init__(self, llm: LLM, cache_size: int = 100):
        """
        Initialize the cached LLM.
        
        Args:
            llm: The underlying LLM
            cache_size: Maximum number of responses to cache
        """
        self.llm = llm
        self.cache = OrderedDict()
        self.cache_size = cache_size
    
    def _hash_prompt(self, prompt: str) -> str:
        """
        Create a hash of the prompt for use as a cache key.
        
        Args:
            prompt: The prompt to hash
            
        Returns:
            A hash of the prompt
        """
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response to a prompt, using cached values when available.
        
        Args:
            prompt: The prompt to generate a response for
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            The generated response
        """
        # Create a cache key from the prompt and kwargs
        key_parts = [prompt]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        key = self._hash_prompt("".join(key_parts))
        
        # Check cache
        if key in self.cache:
            # Move the item to the end of the OrderedDict to mark it as recently used
            self.cache.move_to_end(key)
            logger.info("LLM cache hit")
            return self.cache[key]
        
        # Generate response
        logger.info("LLM cache miss, generating response")
        start_time = time.time()
        response = self.llm.generate(prompt, **kwargs)
        end_time = time.time()
        logger.info(f"LLM response generated in {end_time - start_time:.2f} seconds")
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = response
        
        return response


class CachedRetriever:
    """
    A wrapper around a retriever that caches results to avoid redundant computation.
    """
    
    def __init__(self, retriever: BaseRetriever, cache_size: int = 100):
        """
        Initialize the cached retriever.
        
        Args:
            retriever: The underlying retriever
            cache_size: Maximum number of queries to cache
        """
        self.retriever = retriever
        self.cache = OrderedDict()
        self.cache_size = cache_size
    
    def _hash_query(self, query: str) -> str:
        """
        Create a hash of the query for use as a cache key.
        
        Args:
            query: The query to hash
            
        Returns:
            A hash of the query
        """
        return hashlib.md5(query.encode()).hexdigest()
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for a query, using cached values when available.
        
        Args:
            query: The query to search for
            
        Returns:
            A list of relevant documents
        """
        key = self._hash_query(query)
        
        # Check cache
        if key in self.cache:
            # Move the item to the end of the OrderedDict to mark it as recently used
            self.cache.move_to_end(key)
            logger.info("Retriever cache hit")
            return self.cache[key]
        
        # Retrieve documents
        logger.info("Retriever cache miss, fetching documents")
        start_time = time.time()
        docs = self.retriever.get_relevant_documents(query)
        end_time = time.time()
        logger.info(f"Retrieved {len(docs)} documents in {end_time - start_time:.2f} seconds")
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = docs
        
        return docs


# ============================================================================
# Asynchronous Processing Implementations
# ============================================================================

class AsyncRetriever:
    """
    A wrapper around multiple retrievers that queries them asynchronously.
    """
    
    def __init__(self, retrievers: Dict[str, BaseRetriever]):
        """
        Initialize the async retriever.
        
        Args:
            retrievers: A dictionary mapping source names to retrievers
        """
        self.retrievers = retrievers
    
    async def _get_documents_async(self, source: str, retriever: BaseRetriever, query: str) -> List[Document]:
        """
        Get documents from a single retriever asynchronously.
        
        Args:
            source: The name of the source
            retriever: The retriever to query
            query: The query to search for
            
        Returns:
            A list of documents with source metadata
        """
        # Run the retriever in a thread pool
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None, lambda: retriever.get_relevant_documents(query)
        )
        
        # Add source metadata to each document
        for doc in docs:
            doc.metadata["source"] = source
        
        return docs
    
    async def get_relevant_documents_async(self, query: str) -> Dict[str, List[Document]]:
        """
        Get relevant documents from all retrievers asynchronously.
        
        Args:
            query: The query to search for
            
        Returns:
            A dictionary mapping source names to lists of documents
        """
        # Create tasks for each retriever
        tasks = []
        for source, retriever in self.retrievers.items():
            task = self._get_documents_async(source, retriever, query)
            tasks.append((source, task))
        
        # Wait for all tasks to complete
        results = {}
        for source, task in tasks:
            docs = await task
            results[source] = docs
        
        return results


# ============================================================================
# Model Optimization Implementations
# ============================================================================

def get_optimized_llm(model_name: str, quantization_level: str = "4bit"):
    """
    Get an optimized LLM based on the specified quantization level.
    
    Args:
        model_name: The name of the model to load
        quantization_level: The quantization level to use (4bit, 8bit, or none)
        
    Returns:
        An initialized LLM
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_community.llms import HuggingFacePipeline
    
    # Configure quantization if GPU is available
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using {quantization_level} quantization.")
        
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
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.2,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Create LangChain pipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm


# ============================================================================
# Integration Example
# ============================================================================

class OptimizedLiquibaseReviewer:
    """
    Optimized reviewer for Liquibase migrations using caching and async processing.
    """
    
    def __init__(self, quantization_level: str = "4bit"):
        """
        Initialize the optimized reviewer.
        
        Args:
            quantization_level: The quantization level to use for the LLM
        """
        from backend.models.liquibase_parser import LiquibaseParser
        from backend.models.vector_store import get_retriever, get_embedding_model
        
        # Initialize parser
        self.parser = LiquibaseParser()
        
        # Initialize embedding model with caching
        embedding_model = get_embedding_model()
        self.cached_embeddings = CachedEmbeddings(embedding_model, cache_size=1000)
        
        # Initialize LLM with optimization and caching
        self.llm = get_optimized_llm("mistral-7b-instruct-v0.2.Q4_K_M.gguf", quantization_level)
        self.cached_llm = CachedLLM(self.llm, cache_size=100)
        
        # Get retrievers for different document categories with caching
        self.retrievers = {
            "internal_guidelines": CachedRetriever(
                get_retriever(collection_name="internal_guidelines", embedding_model=self.cached_embeddings)
            ),
            "example_migrations": CachedRetriever(
                get_retriever(collection_name="example_migrations", embedding_model=self.cached_embeddings)
            ),
            "liquibase_docs": CachedRetriever(
                get_retriever(collection_name="liquibase_docs", embedding_model=self.cached_embeddings)
            )
        }
        
        # Initialize async retriever
        self.async_retriever = AsyncRetriever(self.retrievers)
    
    async def review_migration_async(self, migration_content: str, format_type: str = "xml") -> str:
        """
        Review a Liquibase migration asynchronously.
        
        Args:
            migration_content: The content of the migration file.
            format_type: The format of the migration file (xml or yaml).
        
        Returns:
            A review of the migration.
        """
        logger.info(f"Reviewing {format_type} migration")
        
        # Parse the migration
        parsed_migration = self._parse_migration_content(migration_content, format_type)
        
        # Extract key elements from the migration
        table_names = self._extract_table_names(parsed_migration)
        change_types = self._extract_change_types(parsed_migration)
        
        # Create targeted queries
        general_query = f"Liquibase migration with {', '.join(change_types)} on tables {', '.join(table_names)}"
        
        # Get relevant documents asynchronously
        docs_by_source = await self.async_retriever.get_relevant_documents_async(general_query)
        
        # Extract content from documents
        content_by_source = {}
        for source, docs in docs_by_source.items():
            content_by_source[source] = [doc.page_content for doc in docs]
        
        # Combine context from different sources
        context = self._combine_context(content_by_source)
        
        # Generate the review
        review = self.cached_llm.generate(
            self._create_review_prompt(
                migration_content=migration_content,
                format_type=format_type,
                parsed_migration=str(parsed_migration),
                context=context
            )
        )
        
        return review
    
    def review_migration(self, migration_content: str, format_type: str = "xml") -> str:
        """
        Review a Liquibase migration (synchronous wrapper around async method).
        
        Args:
            migration_content: The content of the migration file.
            format_type: The format of the migration file (xml or yaml).
        
        Returns:
            A review of the migration.
        """
        # Run the async method in an event loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.review_migration_async(migration_content, format_type)
        )
    
    def _combine_context(self, docs_by_source: Dict[str, List[str]]) -> str:
        """
        Combine context from different sources in priority order.
        
        Args:
            docs_by_source: A dictionary mapping source names to lists of document contents
            
        Returns:
            Combined context
        """
        context_parts = []
        
        # Add internal guidelines (highest priority)
        if "internal_guidelines" in docs_by_source and docs_by_source["internal_guidelines"]:
            context_parts.append("## Internal Guidelines (Highest Priority)\n\n" + 
                                "\n\n".join(docs_by_source["internal_guidelines"]))
        
        # Add example migrations (high priority)
        if "example_migrations" in docs_by_source and docs_by_source["example_migrations"]:
            context_parts.append("## Example Migrations (High Priority)\n\n" + 
                                "\n\n".join(docs_by_source["example_migrations"]))
        
        # Add Liquibase documentation (medium priority)
        if "liquibase_docs" in docs_by_source and docs_by_source["liquibase_docs"]:
            context_parts.append("## Liquibase Documentation (Medium Priority)\n\n" + 
                                "\n\n".join(docs_by_source["liquibase_docs"]))
        
        return "\n\n".join(context_parts)
    
    def _create_review_prompt(self, migration_content: str, format_type: str, parsed_migration: str, context: str) -> str:
        """
        Create a prompt for reviewing a migration.
        
        Args:
            migration_content: The content of the migration file.
            format_type: The format of the migration file (xml or yaml).
            parsed_migration: The parsed migration structure.
            context: The context from relevant documents.
            
        Returns:
            A prompt for reviewing the migration.
        """
        return f"""
        You are a Liquibase migration reviewer. Your task is to review a Liquibase migration against best practices and company guidelines.
        
        # Priority Order for Information Sources
        When reviewing, prioritize information in this order:
        1. Internal Guidelines (highest priority)
        2. Example Migrations (YAML and XML)
        3. Liquibase Documentation (lowest priority)
        
        Only fall back to lower priority sources if higher priority sources don't contain relevant information.
        
        # Migration to Review
        ```{format_type}
        {migration_content}
        ```
        
        # Parsed Migration Structure
        ```
        {parsed_migration}
        ```
        
        # Reference Documentation and Guidelines
        {context}
        
        Please provide a detailed review of the migration, including:
        
        1. **Summary**: A brief summary of what the migration does.
        2. **Compliance**: Does the migration comply with Liquibase best practices and company guidelines?
        3. **Issues**: Identify any issues or potential problems with the migration.
        4. **Recommendations**: Provide specific recommendations for improving the migration.
        5. **Best Practices**: Highlight any best practices that should be followed.
        
        Format your review in Markdown with clear sections and bullet points where appropriate.
        """
