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
        self.llm = get_llm()
        
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
        
        # Parse the migration
        parsed_migration = self._parse_migration_content(migration_content, format_type)
        
        # Get relevant documents from different categories
        liquibase_docs = self._get_relevant_liquibase_docs(parsed_migration)
        internal_guidelines = self._get_relevant_internal_guidelines(parsed_migration)
        example_migrations = self._get_relevant_example_migrations(parsed_migration)
        
        # Combine all relevant documents
        context = self._combine_context(liquibase_docs, internal_guidelines, example_migrations)
        
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
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a Liquibase migration reviewer. Your task is to review a Liquibase migration against best practices and company guidelines.
        
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
