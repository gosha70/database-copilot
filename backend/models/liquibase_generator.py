"""
Liquibase migration generator using RAG.
"""
import logging
from typing import Dict, List, Optional, Union, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import NUM_RETRIEVAL_RESULTS
from backend.models.llm import get_llm
from backend.models.vector_store import get_retriever

logger = logging.getLogger(__name__)

class LiquibaseGenerator:
    """
    Generator for Liquibase migrations using RAG.
    """
    
    def __init__(self):
        """
        Initialize the generator.
        """
        self.llm = get_llm()
        
        # Get retrievers for different document categories
        self.liquibase_docs_retriever = get_retriever(collection_name="liquibase_docs")
        self.internal_guidelines_retriever = get_retriever(collection_name="internal_guidelines")
        self.example_migrations_retriever = get_retriever(collection_name="example_migrations")
    
    def generate_migration(
        self,
        description: str,
        format_type: str = "xml",
        author: str = "database-copilot"
    ) -> str:
        """
        Generate a Liquibase migration from a natural language description.
        
        Args:
            description: Natural language description of the migration.
            format_type: The format of the migration file (xml or yaml).
            author: The author of the migration.
        
        Returns:
            A Liquibase migration.
        """
        logger.info(f"Generating {format_type} migration from description: {description}")
        
        # Get relevant documents from different categories
        liquibase_docs = self._get_relevant_liquibase_docs(description)
        internal_guidelines = self._get_relevant_internal_guidelines(description)
        example_migrations = self._get_relevant_example_migrations(description, format_type)
        
        # Combine all relevant documents
        context = self._combine_context(liquibase_docs, internal_guidelines, example_migrations)
        
        # Create the generation chain
        generation_chain = self._create_generation_chain()
        
        # Generate the migration
        migration = generation_chain.invoke({
            "description": description,
            "format_type": format_type,
            "author": author,
            "context": context
        })
        
        return migration
    
    def _get_relevant_liquibase_docs(self, description: str) -> List[str]:
        """
        Get relevant Liquibase documentation.
        
        Args:
            description: Natural language description of the migration.
        
        Returns:
            A list of relevant Liquibase documentation.
        """
        # Create a query based on the description
        query = f"Liquibase documentation for: {description}"
        
        # Get relevant documents
        docs = self.liquibase_docs_retriever.get_relevant_documents(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _get_relevant_internal_guidelines(self, description: str) -> List[str]:
        """
        Get relevant internal guidelines.
        
        Args:
            description: Natural language description of the migration.
        
        Returns:
            A list of relevant internal guidelines.
        """
        # Create a query based on the description
        query = f"Internal guidelines for database migrations: {description}"
        
        # Get relevant documents
        docs = self.internal_guidelines_retriever.get_relevant_documents(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _get_relevant_example_migrations(self, description: str, format_type: str) -> List[str]:
        """
        Get relevant example migrations.
        
        Args:
            description: Natural language description of the migration.
            format_type: The format of the migration file (xml or yaml).
        
        Returns:
            A list of relevant example migrations.
        """
        # Create a query based on the description and format type
        query = f"Example {format_type} migrations for: {description}"
        
        # Get relevant documents
        docs = self.example_migrations_retriever.get_relevant_documents(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _combine_context(
        self,
        liquibase_docs: List[str],
        internal_guidelines: List[str],
        example_migrations: List[str]
    ) -> str:
        """
        Combine context from different sources.
        
        Args:
            liquibase_docs: Relevant Liquibase documentation.
            internal_guidelines: Relevant internal guidelines.
            example_migrations: Relevant example migrations.
        
        Returns:
            Combined context.
        """
        context_parts = []
        
        # Add Liquibase documentation
        if liquibase_docs:
            context_parts.append("## Liquibase Documentation\n\n" + "\n\n".join(liquibase_docs))
        
        # Add internal guidelines
        if internal_guidelines:
            context_parts.append("## Internal Guidelines\n\n" + "\n\n".join(internal_guidelines))
        
        # Add example migrations
        if example_migrations:
            context_parts.append("## Example Migrations\n\n" + "\n\n".join(example_migrations))
        
        return "\n\n".join(context_parts)
    
    def _create_generation_chain(self):
        """
        Create a chain for generating migrations.
        
        Returns:
            A chain for generating migrations.
        """
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a Liquibase migration generator. Your task is to generate a Liquibase migration based on a natural language description.
        
        # Migration Description
        {description}
        
        # Format Type
        {format_type}
        
        # Author
        {author}
        
        # Reference Documentation and Guidelines
        {context}
        
        Please generate a complete and valid Liquibase migration in {format_type} format based on the description.
        
        Follow these guidelines:
        1. Use a meaningful changeset ID and include the author.
        2. Include appropriate comments to explain the purpose of the migration.
        3. Follow Liquibase best practices and company guidelines.
        4. Ensure the migration is complete and ready to be executed.
        5. Include all necessary attributes and constraints.
        
        Return ONLY the migration content without any additional explanation.
        """)
        
        # Create the chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
