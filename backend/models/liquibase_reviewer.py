"""
Liquibase migration reviewer using RAG.
"""
import logging
from typing import Dict, List, Optional, Union, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import NUM_RETRIEVAL_RESULTS
from backend.models.llm import get_llm
from backend.models.vector_store import get_retriever
from backend.models.liquibase_parser import LiquibaseParser

logger = logging.getLogger(__name__)

class LiquibaseReviewer:
    """
    Reviewer for Liquibase migrations using RAG.
    """
    
    def __init__(self):
        """
        Initialize the reviewer.
        """
        self.parser = LiquibaseParser()
        
        # Import at the module level to avoid circular imports
        from backend.models.streamlit_compatibility import get_safe_llm
        from backend.config import LLM_TYPE
        
        # Use get_safe_llm instead of get_llm to avoid dependency issues
        try:
            # Try to use an external LLM if configured
            if LLM_TYPE != "local":
                self.llm = get_safe_llm(use_external=True)
            else:
                self.llm = get_safe_llm()
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise e
        
        # Get retrievers for different document categories
        self.liquibase_docs_retriever = get_retriever(collection_name="liquibase_docs")
        self.internal_guidelines_retriever = get_retriever(collection_name="internal_guidelines")
        self.example_migrations_retriever = get_retriever(collection_name="example_migrations")
    
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
        
        try:
            # Log which LLM is being used
            if hasattr(self.llm, 'is_external_llm') and self.llm.is_external_llm:
                logger.info(f"Using external LLM: {self.llm.provider_name} with model {self.llm.model_name}")
            else:
                logger.info(f"Using local LLM: {getattr(self.llm, '_llm_type', 'unknown')}")
            
            # Parse the migration
            parsed_migration = self._parse_migration_content(migration_content, format_type)
            if not parsed_migration or (isinstance(parsed_migration, dict) and "error" in parsed_migration):
                error_msg = parsed_migration.get("error", "Unknown parsing error") if isinstance(parsed_migration, dict) else "Failed to parse migration"
                logger.error(f"Error parsing migration: {error_msg}")
                return f"Error: Failed to parse the migration file. {error_msg}"
            
            # Get relevant documents from different categories
            try:
                liquibase_docs = self._get_relevant_liquibase_docs(parsed_migration)
                logger.info(f"Retrieved {len(liquibase_docs)} Liquibase docs")
            except Exception as e:
                logger.error(f"Error retrieving Liquibase docs: {e}")
                liquibase_docs = []
            
            try:
                internal_guidelines = self._get_relevant_internal_guidelines(parsed_migration)
                logger.info(f"Retrieved {len(internal_guidelines)} internal guidelines")
            except Exception as e:
                logger.error(f"Error retrieving internal guidelines: {e}")
                internal_guidelines = []
            
            try:
                example_migrations = self._get_relevant_example_migrations(parsed_migration)
                logger.info(f"Retrieved {len(example_migrations)} example migrations")
            except Exception as e:
                logger.error(f"Error retrieving example migrations: {e}")
                example_migrations = []
            
            # Check if we have any context
            if not liquibase_docs and not internal_guidelines and not example_migrations:
                logger.warning("No context retrieved from any source")
                # Continue with empty context, but log a warning
            
            # Combine all relevant documents
            context = self._combine_context(liquibase_docs, internal_guidelines, example_migrations)
            
            # Create the review chain
            review_chain = self._create_review_chain()
            
            # Generate the review
            try:
                # Set up a synchronous environment for the LLM call
                import asyncio
                
                # Check if we're in an event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in a running event loop, use it
                        logger.info("Using existing event loop")
                    else:
                        # Loop exists but isn't running, create a new one
                        logger.info("Creating new event loop")
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                except RuntimeError:
                    # No event loop exists, create a new one
                    logger.info("No event loop found, creating new one")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the LLM call
                review = review_chain.invoke({
                    "migration_content": migration_content,
                    "format_type": format_type,
                    "parsed_migration": str(parsed_migration),
                    "context": context
                })
                
                # Check if the review contains error messages
                if review.startswith("ERROR:") or "This is a placeholder response from a fallback system" in review:
                    logger.error(f"LLM returned an error: {review}")
                    return f"Error: The LLM returned an error response. Please check the logs for details."
                
                # Check if the review contains the DO NOT instructions that should have been removed
                if "DO NOT include" in review:
                    logger.warning("Review contains prompt instructions that should have been removed")
                    # Clean up the review by removing the DO NOT instructions
                    lines = review.split('\n')
                    cleaned_lines = [line for line in lines if not line.strip().startswith("DO NOT")]
                    review = '\n'.join(cleaned_lines)
                
                return review
                
            except Exception as e:
                logger.error(f"Error generating review: {e}")
                return f"Error: Failed to generate review. {str(e)}"
            
        except Exception as e:
            logger.error(f"Error in review_migration: {e}")
            return f"Error: An unexpected error occurred during the review process. {str(e)}"
    
    def _parse_migration_content(self, migration_content: str, format_type: str) -> Dict[str, Any]:
        """
        Parse migration content.
        
        Args:
            migration_content: The content of the migration file.
            format_type: The format of the migration file (xml or yaml).
        
        Returns:
            A dictionary containing the parsed migration.
        """
        # Create a temporary file to parse
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=f".{format_type}", delete=False) as temp_file:
            temp_file.write(migration_content.encode())
            temp_file_path = temp_file.name
        
        try:
            # Parse the file
            parsed_migration = self.parser.parse_file(temp_file_path)
            return parsed_migration
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    
    def _get_relevant_liquibase_docs(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Get relevant Liquibase documentation.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of relevant Liquibase documentation.
        """
        # Extract change types from the parsed migration
        change_types = self._extract_change_types(parsed_migration)
        
        # Create a query based on the change types
        query = f"Liquibase documentation for: {', '.join(change_types)}"
        
        # Get relevant documents
        docs = self.liquibase_docs_retriever.invoke(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _get_relevant_internal_guidelines(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Get relevant internal guidelines.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of relevant internal guidelines.
        """
        # Extract table names and change types from the parsed migration
        table_names = self._extract_table_names(parsed_migration)
        change_types = self._extract_change_types(parsed_migration)
        
        # Create a query based on the table names and change types
        query = f"Internal guidelines for database migrations with: {', '.join(change_types)} on tables: {', '.join(table_names)}"
        
        # Get relevant documents
        docs = self.internal_guidelines_retriever.invoke(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _get_relevant_example_migrations(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Get relevant example migrations.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of relevant example migrations.
        """
        # Extract table names and change types from the parsed migration
        table_names = self._extract_table_names(parsed_migration)
        change_types = self._extract_change_types(parsed_migration)
        
        # Create a query based on the table names and change types
        query = f"Example migrations with: {', '.join(change_types)} on tables: {', '.join(table_names)}"
        
        # Get relevant documents
        docs = self.example_migrations_retriever.invoke(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _extract_change_types(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Extract change types from a parsed migration.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of change types.
        """
        change_types = set()
        
        # Extract change types from changesets
        if 'databaseChangeLog' in parsed_migration:
            # Handle both list and dict formats for databaseChangeLog
            changeset_list = []
            if isinstance(parsed_migration['databaseChangeLog'], dict):
                changeset_list = parsed_migration['databaseChangeLog'].get('changeSet', [])
            elif isinstance(parsed_migration['databaseChangeLog'], list):
                # If databaseChangeLog is a list, find the changeSet entries
                for item in parsed_migration['databaseChangeLog']:
                    if isinstance(item, dict) and 'changeSet' in item:
                        if isinstance(item['changeSet'], list):
                            changeset_list.extend(item['changeSet'])
                        else:
                            changeset_list.append(item['changeSet'])
            
            # Process each changeset
            for changeset in changeset_list:
                # Handle both list and dict formats for changes
                changes_list = []
                if isinstance(changeset, dict) and 'changes' in changeset:
                    changes = changeset['changes']
                    if isinstance(changes, list):
                        changes_list = changes
                    elif isinstance(changes, dict):
                        changes_list = [changes]
                
                # Extract change types
                for change in changes_list:
                    if isinstance(change, dict):
                        change_types.update(change.keys())
        
        return list(change_types) if change_types else ["unknown"]
    
    def _extract_table_names(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Extract table names from a parsed migration.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of table names.
        """
        table_names = set()
        
        # Extract table names from changesets
        if 'databaseChangeLog' in parsed_migration:
            # Handle both list and dict formats for databaseChangeLog
            changeset_list = []
            if isinstance(parsed_migration['databaseChangeLog'], dict):
                changeset_list = parsed_migration['databaseChangeLog'].get('changeSet', [])
            elif isinstance(parsed_migration['databaseChangeLog'], list):
                # If databaseChangeLog is a list, find the changeSet entries
                for item in parsed_migration['databaseChangeLog']:
                    if isinstance(item, dict) and 'changeSet' in item:
                        if isinstance(item['changeSet'], list):
                            changeset_list.extend(item['changeSet'])
                        else:
                            changeset_list.append(item['changeSet'])
            
            # Process each changeset
            for changeset in changeset_list:
                # Handle both list and dict formats for changes
                changes_list = []
                if isinstance(changeset, dict) and 'changes' in changeset:
                    changes = changeset['changes']
                    if isinstance(changes, list):
                        changes_list = changes
                    elif isinstance(changes, dict):
                        changes_list = [changes]
                
                # Extract table names from each change
                for change in changes_list:
                    if not isinstance(change, dict):
                        continue
                    
                    for change_type, change_data in change.items():
                        if not isinstance(change_data, dict):
                            continue
                            
                        # Extract table name based on change type
                        if change_type == 'createTable' and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
                        elif change_type == 'addColumn' and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
                        elif change_type == 'dropTable' and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
                        elif change_type == 'addForeignKeyConstraint':
                            if 'baseTableName' in change_data:
                                table_names.add(change_data['baseTableName'])
                            if 'referencedTableName' in change_data:
                                table_names.add(change_data['referencedTableName'])
                        elif change_type == 'addCheckConstraint' and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
                        elif change_type in ['addPrimaryKey', 'addUniqueConstraint', 'createIndex'] and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
        
        return list(table_names) if table_names else ["unknown"]
    
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
    
    def _create_review_chain(self):
        """
        Create a chain for reviewing migrations.
        
        Returns:
            A chain for reviewing migrations.
        """
        # Create the prompt template with detailed requirements
        prompt = ChatPromptTemplate.from_template("""
        You are a Liquibase migration reviewer. Your task is to provide a detailed, technical review of a Liquibase migration against best practices and company guidelines.

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
        
        Analyze the migration and provide a detailed review with the following sections:
        
        ## Summary
        [Provide a brief description of what the migration does, including tables created/modified, constraints added, and other significant changes.]
        
        ## Compliance
        [Evaluate compliance with Liquibase best practices and company guidelines, including:]
        - One change type per changeset: [Compliant/Non-compliant]
        - Proper author and ID attributes: [Compliant/Non-compliant]
        - Rollback sections for each changeset: [Compliant/Non-compliant]
        - Proper naming conventions: [Compliant/Non-compliant]
        - Appropriate data types and lengths: [Compliant/Non-compliant]
        - Required constraints: [Compliant/Non-compliant]
        
        ## Issues
        [List specific issues with the migration, including exact names, IDs, and line numbers. If no issues are found in a category, state "No issues found."]
        
        ### Naming Convention Violations
        [List specific table, column, constraint names that violate conventions]
        
        ### Missing Rollback Sections
        [List specific changesets missing rollback sections]
        
        ### Data Type Concerns
        [List columns with inappropriate types]
        
        ### Missing Constraints
        [List columns missing required constraints]
        
        ### Performance Concerns
        [List any performance issues like large tables without indexes]
        
        ### Data Loss Risks
        [List operations that could cause data loss]
        
        ### Incorrect Liquibase Usage
        [List any incorrect usage of Liquibase commands or syntax]
        
        ## Recommendations
        [Provide specific, actionable recommendations with exact code snippets showing how to fix each issue]
        
        ## Best Practices
        [Highlight best practices that are followed or should be followed]
        
        IMPORTANT: Your review must be based on the actual content of the migration. Do not use placeholders in your final response. Replace all bracketed sections with actual analysis. If you cannot determine something, say so explicitly rather than using a placeholder.
        
        You must:
        1. Reference actual code from the migration (exact table names, column names, constraint names, etc.)
        2. Include specific changeset IDs when identifying issues
        3. Provide concrete, copy-pastable code examples in your recommendations
        4. Be technically precise and actionable
        5. Focus on database design, Liquibase usage, and SQL best practices
        
        Format your review in Markdown with clear sections and bullet points where appropriate.
        """)
        
        # Create the chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
