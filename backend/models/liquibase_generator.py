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
    
    def __init__(self, debug_mode=False):
        """
        Initialize the generator.
        
        Args:
            debug_mode: Whether to enable debug mode, which logs more information.
        """
        self.llm = get_llm()
        self.debug_mode = debug_mode
        
        if self.debug_mode:
            logger.info("Debug mode enabled for LiquibaseGenerator")
    
    def generate_migration(
        self,
        description: str,
        format_type: str = "xml",
        author: str = "database-copilot"
    ) -> str:
        """
        Generate a Liquibase migration from a natural language description.
        
        This method creates new retrievers for each request to avoid context reuse
        between different generation requests. This ensures that each migration
        is generated based only on the current description, not influenced by
        previous requests.
        
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
        
        # Create a new generation chain for each request to avoid context reuse
        generation_chain = self._create_generation_chain()
        
        # Log the context if debug mode is enabled
        if self.debug_mode:
            logger.info("=== DEBUG: RETRIEVED CONTEXT FOR MIGRATION GENERATION ===")
            logger.info(f"Description: {description}")
            logger.info(f"Format: {format_type}")
            logger.info(f"Author: {author}")
            logger.info(f"Context length: {len(context)} characters")
            logger.info("=== Liquibase Docs ===")
            for i, doc in enumerate(liquibase_docs):
                logger.info(f"Doc {i+1} (length: {len(doc)} chars): {doc[:200]}...")
            logger.info("=== Internal Guidelines ===")
            for i, doc in enumerate(internal_guidelines):
                logger.info(f"Doc {i+1} (length: {len(doc)} chars): {doc[:200]}...")
            logger.info("=== Example Migrations ===")
            for i, doc in enumerate(example_migrations):
                logger.info(f"Doc {i+1} (length: {len(doc)} chars): {doc[:200]}...")
            logger.info("=== END DEBUG: RETRIEVED CONTEXT ===")
        
        # Generate the migration
        migration = generation_chain.invoke({
            "description": description,
            "format_type": format_type,
            "author": author,
            "context": context
        })
        
        # Log the raw migration if debug mode is enabled
        if self.debug_mode:
            logger.info("=== DEBUG: RAW MIGRATION BEFORE POST-PROCESSING ===")
            logger.info(migration)
            logger.info("=== END DEBUG: RAW MIGRATION ===")
        
        # Post-process the migration to remove any "Example:" text
        migration = self._post_process_migration(migration)
        
        return migration
    
    def _post_process_migration(self, migration: str) -> str:
        """
        Post-process the generated migration to remove any unwanted text.
        
        Args:
            migration: The generated migration.
        
        Returns:
            The post-processed migration.
        """
        # Remove any "Example:" text
        lines = migration.split('\n')
        cleaned_lines = []
        skip_next_line = False
        skip_section = False
        instruction_count = 0
        
        # Check for common instruction patterns
        instruction_patterns = [
            "ensure that the generated migration",
            "if you're generating",
            "make sure",
            "remember to",
            "don't forget",
            "be sure to",
            "it's important to",
            "note that",
            "please note",
            "keep in mind",
            "consider",
            "you should",
            "you must",
            "you need to",
            "you'll want to",
            "you'll need to",
            "you may want to",
            "you may need to",
            "you might want to",
            "you might need to"
        ]
        
        # Check for numbered instructions
        numbered_instruction_pattern = r'^\s*\d+\.\s+'
        import re
        
        for line in lines:
            # Skip lines that contain "Example:" or similar text
            if "example:" in line.lower() or ("example" in line.lower() and ":" in line):
                skip_next_line = True
                continue
            
            # Skip separator lines that follow "Example:" lines
            if skip_next_line and (line.strip() == '' or all(c == '-' for c in line.strip()) or all(c == '=' for c in line.strip())):
                skip_next_line = False
                continue
            
            # Check if line contains instruction patterns
            is_instruction = False
            for pattern in instruction_patterns:
                if pattern in line.lower():
                    is_instruction = True
                    instruction_count += 1
                    break
            
            # Check if line is a numbered instruction
            if re.match(numbered_instruction_pattern, line):
                is_instruction = True
                instruction_count += 1
            
            # Skip instruction lines
            if is_instruction:
                skip_section = True
                continue
            
            # Skip empty lines after instructions
            if skip_section and not line.strip():
                continue
            
            # Reset skip_section on a line that looks like actual migration content
            if skip_section and (line.strip().startswith("--") or 
                                line.strip().startswith("<") or 
                                line.strip().startswith("databaseChangeLog:") or
                                "changeSet" in line or
                                "createTable" in line or
                                "addColumn" in line):
                skip_section = False
            
            # Reset the skip flag
            skip_next_line = False
            
            # Skip the line if we're in a section to skip
            if skip_section:
                continue
            
            # Add the line to the cleaned lines
            cleaned_lines.append(line)
        
        # Join the cleaned lines
        cleaned_migration = '\n'.join(cleaned_lines)
        
        # Remove any leading/trailing whitespace
        cleaned_migration = cleaned_migration.strip()
        
        # If we removed a lot of instructions, log a warning
        if instruction_count > 5:
            logger.warning(f"Removed {instruction_count} instruction lines from migration")
        
        # If the migration is empty after cleaning, return an error message
        if not cleaned_migration:
            logger.error("Migration is empty after post-processing")
            return "Error: The LLM returned instructions instead of a migration. Please try again."
        
        # Check if the migration looks valid
        valid_markers = ["changeSet", "createTable", "addColumn", "databaseChangeLog:", "<?xml", "<databaseChangeLog"]
        is_valid = any(marker in cleaned_migration for marker in valid_markers)
        
        if not is_valid:
            logger.error("Migration does not contain valid Liquibase syntax")
            return """Error: The LLM returned invalid content instead of a migration.

Please try again with a more specific description, for example:
"Create a Liquibase migration to add a table named 'customer' with columns for id (primary key), firstName, and lastName"
"""
        
        return cleaned_migration
    
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
        
        # Create a new retriever for each request to avoid context reuse
        retriever = get_retriever(collection_name="liquibase_docs")
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        
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
        
        # Create a new retriever for each request to avoid context reuse
        retriever = get_retriever(collection_name="internal_guidelines")
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        
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
        
        # Create a new retriever for each request to avoid context reuse
        retriever = get_retriever(collection_name="example_migrations")
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        
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
        Generate a Liquibase migration in {format_type} format for the following description:

        {description}

        Author: {author}

        IMPORTANT: Return ONLY the migration code, no explanations or instructions.

        For XML format, include proper XML headers and namespace declarations.
        For YAML format, ensure proper indentation and structure.

        Use BIGINT for long IDs and VARCHAR(255) for string fields.
        Include appropriate constraints (primary key, nullable, etc.).
        """)
        
        # Create the chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
