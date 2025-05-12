"""
Example implementation of the CascadeRetriever class for the Database Copilot.

This file demonstrates how to implement the cascade retrieval system
described in the enhancement plan. It's meant as a reference implementation
and should be adapted to fit into the actual codebase.
"""
import logging
from typing import Dict, List, Optional, Union, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class CascadeRetriever(BaseRetriever):
    """
    A retriever that cascades through multiple retrievers in priority order.
    
    This retriever queries multiple sources in a priority order, only falling back
    to lower-priority sources if higher-priority sources don't return enough results.
    """
    
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
        self.retrievers = retrievers
        self.priority_order = priority_order
        self.min_docs_per_source = min_docs_per_source
        self.max_docs_total = max_docs_total
        
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
        for source in self.priority_order:
            # Skip if we already have enough documents
            if total_docs >= self.max_docs_total:
                break
                
            # Get the retriever for this source
            retriever = self.retrievers[source]
            
            # Calculate how many documents we still need
            docs_needed = min(self.min_docs_per_source, self.max_docs_total - total_docs)
            
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
                    doc.metadata["priority"] = self.priority_order.index(source)
            
            # Add documents to the result
            all_docs.extend(docs[:docs_needed])
            total_docs += len(docs[:docs_needed])
            
            logger.info(f"Got {len(docs[:docs_needed])} documents from source '{source}'")
        
        # Sort documents by priority
        all_docs.sort(key=lambda doc: doc.metadata.get("priority", 999))
        
        return all_docs[:self.max_docs_total]


# Example usage:

def create_cascade_retriever(
    internal_guidelines_retriever,
    example_migrations_retriever,
    liquibase_docs_retriever,
    min_docs_per_source=3,
    max_docs_total=10
):
    """
    Create a cascade retriever for the Liquibase reviewer.
    
    Args:
        internal_guidelines_retriever: Retriever for internal guidelines
        example_migrations_retriever: Retriever for example migrations
        liquibase_docs_retriever: Retriever for Liquibase documentation
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
    
    # Define the priority order
    priority_order = [
        "internal_guidelines",  # Highest priority
        "example_migrations",   # Medium priority
        "liquibase_docs"        # Lowest priority
    ]
    
    # Create and return the cascade retriever
    return CascadeRetriever(
        retrievers=retrievers,
        priority_order=priority_order,
        min_docs_per_source=min_docs_per_source,
        max_docs_total=max_docs_total
    )


# Example integration with LiquibaseReviewer:

class EnhancedLiquibaseReviewer:
    """
    Enhanced reviewer for Liquibase migrations using cascading RAG.
    """
    
    def __init__(self):
        """
        Initialize the reviewer.
        """
        self.parser = LiquibaseParser()
        self.llm = get_llm()
        
        # Get retrievers for different document categories
        self.liquibase_docs_retriever = get_retriever(collection_name="liquibase_docs")
        self.internal_guidelines_retriever = get_retriever(collection_name="internal_guidelines")
        self.example_migrations_retriever = get_retriever(collection_name="example_migrations")
        
        # Create cascade retrievers for different query types
        self.general_cascade_retriever = create_cascade_retriever(
            self.internal_guidelines_retriever,
            self.example_migrations_retriever,
            self.liquibase_docs_retriever
        )
    
    def review_migration(self, migration_content: str, format_type: str = "xml") -> str:
        """
        Review a Liquibase migration.
        
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
        
        # Get relevant documents using cascade retriever
        docs = self.general_cascade_retriever.get_relevant_documents(general_query)
        
        # Organize documents by source
        docs_by_source = {}
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc.page_content)
        
        # Combine context from different sources
        context = self._combine_context(docs_by_source)
        
        # Create the review chain
        review_chain = self._create_review_chain()
        
        # Generate the review
        review = review_chain.invoke({
            "migration_content": migration_content,
            "format_type": format_type,
            "parsed_migration": str(parsed_migration),
            "context": context
        })
        
        return review
    
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
    
    def _create_review_chain(self):
        """
        Create a chain for reviewing migrations.
        
        Returns:
            A chain for reviewing migrations.
        """
        # Create the prompt template with explicit prioritization instructions
        prompt = ChatPromptTemplate.from_template("""
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
        """)
        
        # Create the chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
