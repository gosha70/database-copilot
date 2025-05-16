"""
Cascade retriever implementation for the Database Copilot.

This module provides a cascade retriever that queries multiple sources in priority order,
only falling back to lower-priority sources if higher-priority sources don't return enough results.
"""
import logging
from typing import Dict, List, Optional, Union, Any

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

class CascadeRetriever(BaseRetriever):
    """
    A retriever that cascades through multiple retrievers in priority order.
    
    This retriever queries multiple sources in a priority order, only falling back
    to lower-priority sources if higher-priority sources don't return enough results.
    """
    _retrievers: Dict[str, BaseRetriever] = PrivateAttr()
    _priority_order: List[str] = PrivateAttr()
    _min_docs_per_source: int = PrivateAttr()
    _max_docs_total: int = PrivateAttr()

    
    def __init__(
        self,
        retrievers: Dict[str, BaseRetriever],
        priority_order: List[str],
        min_docs_per_source: int = 3,
        max_docs_total: int = 10
    ):
        """
        Initialize the cascade retriever.
        
        Args:
            retrievers: A dictionary mapping source names to retrievers
            priority_order: A list of source names in priority order (highest first)
            min_docs_per_source: Minimum number of documents to consider sufficient from each source
            max_docs_total: Maximum total number of documents to return
        """
        super().__init__()
        object.__setattr__(self, "_retrievers", retrievers)
        object.__setattr__(self, "_priority_order", priority_order)
        object.__setattr__(self, "_min_docs_per_source", min_docs_per_source)
        object.__setattr__(self, "_max_docs_total", max_docs_total)


        # Validate that all sources in priority_order exist in retrievers
        for source in priority_order:
            if source not in retrievers:
                raise ValueError(f"Source '{source}' in priority_order not found in retrievers")
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents from multiple sources in priority order.
        
        Args:
            query: The query to search for
            
        Returns:
            A list of relevant documents from all sources
        """
        all_docs = []
        total_docs = 0

        # Query each source in priority order
        for source in self._priority_order:
            # Skip if we already have enough documents
            if total_docs >= self._max_docs_total:
                break

            # Get the retriever for this source
            retriever = self._retrievers[source]

            # Calculate how many documents we still need
            docs_needed = min(self._min_docs_per_source, self._max_docs_total - total_docs)

            # Skip if we don't need any more documents
            if docs_needed <= 0:
                continue

            # Get documents from this source
            logger.info(f"Querying source '{source}' for query: {query}")
            docs = retriever.get_relevant_documents(query)

            # Add source metadata to each document
            for doc in docs:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = source
                if "priority" not in doc.metadata:
                    doc.metadata["priority"] = self._priority_order.index(source)

            # Add documents to the result
            all_docs.extend(docs[:docs_needed])
            total_docs += len(docs[:docs_needed])

            logger.info(f"Got {len(docs[:docs_needed])} documents from source '{source}'")

        # Sort documents by priority
        all_docs.sort(key=lambda doc: doc.metadata.get("priority", 999))

        return all_docs[:self._max_docs_total]


def create_cascade_retriever(
    internal_guidelines_retriever,
    example_migrations_retriever,
    liquibase_docs_retriever,
    java_files_retriever=None,
    min_docs_per_source=3,
    max_docs_total=10
):
    """
    Create a cascade retriever for the Liquibase reviewer.
    
    Args:
        internal_guidelines_retriever: Retriever for internal guidelines
        example_migrations_retriever: Retriever for example migrations
        liquibase_docs_retriever: Retriever for Liquibase documentation
        java_files_retriever: Retriever for Java files (optional)
        min_docs_per_source: Minimum number of documents to consider sufficient from each source
        max_docs_total: Maximum total number of documents to return
        
    Returns:
        A cascade retriever
    """
    # Create a dictionary of retrievers
    retrievers = {
        "internal_guidelines": internal_guidelines_retriever,
        "example_migrations": example_migrations_retriever,
        "liquibase_docs": liquibase_docs_retriever
    }
    
    # Add Java files retriever if provided
    if java_files_retriever is not None:
        retrievers["java_files"] = java_files_retriever
    
    # Define the priority order
    priority_order = [
        "example_migrations",   # Highest priority
    ]
    
    # Add Java files to priority order if available
    if "java_files" in retrievers:
        priority_order.append("java_files")  # High priority
    
    # Add other sources in decreasing priority
    priority_order.append("internal_guidelines")  # Medium priority
    priority_order.append("liquibase_docs")  # Low priority
    
    # Create and return the cascade retriever
    return CascadeRetriever(
        retrievers=retrievers,
        priority_order=priority_order,
        min_docs_per_source=min_docs_per_source,
        max_docs_total=max_docs_total
    )
