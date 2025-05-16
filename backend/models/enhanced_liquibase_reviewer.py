"""
Enhanced Liquibase migration reviewer using cascade RAG.

This module provides an enhanced version of the LiquibaseReviewer class
that uses a cascade retriever to prioritize information sources.
"""
import logging
import threading
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import NUM_RETRIEVAL_RESULTS
from backend.models.streamlit_compatibility import get_safe_llm
from backend.models.vector_store import get_retriever
from backend.models.liquibase_parser import LiquibaseParser
from backend.models.cascade_retriever import create_cascade_retriever

logger = logging.getLogger(__name__)

class EnhancedLiquibaseReviewer:
    """
    Enhanced reviewer for Liquibase migrations using cascade RAG.
    """
    
    def __init__(self):
        """
        Initialize the enhanced reviewer.
        """
        self.parser = LiquibaseParser()
        
        # Import at the module level to avoid circular imports
        from backend.models.streamlit_compatibility import get_safe_llm
        from backend.config import get_current_llm_type
        
        # Use get_safe_llm with external LLM if configured
        try:
            # Try to use an external LLM if configured
            current_llm_type = get_current_llm_type()
            if current_llm_type != "local":
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
        
        # Create cascade retriever
        self.cascade_retriever = create_cascade_retriever(
            self.internal_guidelines_retriever,
            self.example_migrations_retriever,
            self.liquibase_docs_retriever,
            min_docs_per_source=3,
            max_docs_total=NUM_RETRIEVAL_RESULTS
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
        
        # Log which LLM is being used
        if hasattr(self.llm, 'is_external_llm') and self.llm.is_external_llm:
            logger.info(f"Using external LLM: {self.llm.provider_name} with model {self.llm.model_name}")
        else:
            logger.info(f"Using local LLM: {getattr(self.llm, '_llm_type', 'unknown')}")
        
        # Parse the migration
        parsed_migration = self._parse_migration_content(migration_content, format_type)
        
        # Extract key elements from the migration
        table_names = self._extract_table_names(parsed_migration)
        change_types = self._extract_change_types(parsed_migration)
        constraints = self._extract_constraints(parsed_migration)
        
        # Create targeted queries
        general_query = f"Liquibase migration with {', '.join(change_types)} on tables {', '.join(table_names)}"
        table_query = f"Database guidelines for tables: {', '.join(table_names)}"
        change_query = f"Migration patterns for: {', '.join(change_types)}"
        constraint_query = f"Constraint guidelines for: {', '.join(constraints)}"
        
        # Get relevant documents using cascade retriever
        # We'll use multiple queries to get more targeted results
        # Use ThreadPoolExecutor to run retrievals in parallel without asyncio
        all_docs = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all retrieval tasks
            general_future = executor.submit(self.cascade_retriever.invoke, general_query)
            table_future = executor.submit(self.cascade_retriever.invoke, table_query)
            change_future = executor.submit(self.cascade_retriever.invoke, change_query)
            constraint_future = executor.submit(self.cascade_retriever.invoke, constraint_query)
            
            # Get results as they complete
            general_docs = general_future.result()
            table_docs = table_future.result()
            change_docs = change_future.result()
            constraint_docs = constraint_future.result()
        
        # Combine all documents
        all_docs = general_docs + table_docs + change_docs + constraint_docs
        
        # Remove duplicates (based on content)
        unique_docs = {}
        for doc in all_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
        
        # Organize documents by source
        docs_by_source = {}
        for doc in unique_docs.values():
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
    
    def _extract_constraints(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Extract constraints from a parsed migration.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of constraint types.
        """
        constraint_types = set()
        
        # Extract constraint types from changesets
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
                
                # Extract constraint types
                for change in changes_list:
                    if not isinstance(change, dict):
                        continue
                    
                    for change_type, change_data in change.items():
                        if not isinstance(change_data, dict):
                            continue
                        
                        # Add constraint types based on change type
                        if change_type in ['addPrimaryKey', 'addForeignKeyConstraint', 'addUniqueConstraint', 'addCheckConstraint']:
                            constraint_types.add(change_type)
                        
                        # Check for column constraints in createTable
                        if change_type == 'createTable' and 'columns' in change_data:
                            columns = change_data['columns']
                            if not isinstance(columns, list):
                                continue
                            
                            for column in columns:
                                if not isinstance(column, dict) or 'column' not in column:
                                    continue
                                
                                column_data = column['column']
                                if not isinstance(column_data, dict) or 'constraints' not in column_data:
                                    continue
                                
                                constraints = column_data['constraints']
                                if not isinstance(constraints, dict):
                                    continue
                                
                                for constraint_type, value in constraints.items():
                                    if value and constraint_type in ['primaryKey', 'foreignKey', 'unique', 'nullable']:
                                        constraint_types.add(constraint_type)
        
        return list(constraint_types) if constraint_types else ["unknown"]
    
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
        You are a Liquibase migration reviewer. Your task is to provide a detailed, technical review of a Liquibase migration against best practices and company guidelines.
        
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
        
        Your review must include the following sections:
        
        1. **Summary**: A brief description of what the migration does, including tables created/modified, constraints added, and other significant changes.
        
        2. **Compliance**: Evaluate compliance with Liquibase best practices and company guidelines:
           - One change type per changeset
           - Proper author and ID attributes
           - Rollback sections for each changeset
           - Proper naming conventions for tables, columns, constraints, and indexes
           - Appropriate data types and lengths
           - Required constraints (NOT NULL, primary keys, etc.)
        
        3. **Issues**: Identify specific issues with the migration, including:
           - Naming convention violations (with exact names that violate conventions)
           - Missing rollback sections (specify which changesets)
           - Data type concerns (specify columns with inappropriate types)
           - Missing constraints (specify which columns should have constraints)
           - Performance concerns (large tables without indexes, etc.)
           - Potential data loss risks
           - Incorrect Liquibase command usage
        
        4. **Recommendations**: Provide specific, actionable recommendations with exact code snippets showing how to fix each issue.
        
        5. **Best Practices**: Highlight any best practices that are followed or should be followed.
        
        IMPORTANT: Your review must:
        - Reference actual code from the migration (exact table names, column names, constraint names, etc.)
        - Include specific line numbers or changeset IDs when identifying issues
        - Provide concrete, copy-pastable code examples in your recommendations
        - Be technically precise and actionable
        - Focus on database design, Liquibase usage, and SQL best practices
        
        Format your review in Markdown with clear sections and bullet points where appropriate.
        
        DO NOT include generic placeholders, signatures, or messages like "Happy reviewing!" or "Remember, your review is a critical part...". Focus ONLY on the technical content of the review.
        
        DO NOT include instructions to yourself in the output. DO NOT include phrases like "DO NOT include personal opinions" or "Focus ONLY on the technical content" in your review. These are instructions for you, not part of the review content.
        """)
        
        # Create the chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
