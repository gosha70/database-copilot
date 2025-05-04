"""
JPA entity generator from Liquibase migrations using RAG.
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

class EntityGenerator:
    """
    Generator for JPA entities from Liquibase migrations using RAG.
    """
    
    def __init__(self):
        """
        Initialize the entity generator.
        """
        self.parser = LiquibaseParser()
        self.llm = get_llm()
        
        # Get retrievers for different document categories
        self.jpa_docs_retriever = get_retriever(collection_name="jpa_docs")
        self.liquibase_docs_retriever = get_retriever(collection_name="liquibase_docs")
        self.internal_guidelines_retriever = get_retriever(collection_name="internal_guidelines")
        self.example_migrations_retriever = get_retriever(collection_name="example_migrations")
    
    def generate_entity(
        self,
        migration_content: str,
        format_type: str = "xml",
        package_name: str = "com.example.entity",
        lombok: bool = True
    ) -> str:
        """
        Generate a JPA entity from a Liquibase migration.
        
        Args:
            migration_content: The content of the migration file.
            format_type: The format of the migration file (xml or yaml).
            package_name: The package name for the generated entity.
            lombok: Whether to use Lombok annotations.
        
        Returns:
            A JPA entity class.
        """
        logger.info(f"Generating JPA entity from {format_type} migration")
        
        # Parse the migration
        parsed_migration = self._parse_migration_content(migration_content, format_type)
        
        # Get relevant documents from different categories
        jpa_docs = self._get_relevant_jpa_docs(parsed_migration)
        liquibase_docs = self._get_relevant_liquibase_docs(parsed_migration)
        internal_guidelines = self._get_relevant_internal_guidelines(parsed_migration)
        example_migrations = self._get_relevant_example_migrations(parsed_migration)
        
        # Combine all relevant documents
        context = self._combine_context(jpa_docs, liquibase_docs, internal_guidelines, example_migrations)
        
        # Create the generation chain
        generation_chain = self._create_generation_chain()
        
        # Generate the entity
        entity = generation_chain.invoke({
            "migration_content": migration_content,
            "format_type": format_type,
            "parsed_migration": str(parsed_migration),
            "package_name": package_name,
            "lombok": lombok,
            "context": context
        })
        
        return entity
    
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
    
    def _get_relevant_jpa_docs(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Get relevant JPA documentation.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of relevant JPA documentation.
        """
        # Extract table names and column types from the parsed migration
        table_names = self._extract_table_names(parsed_migration)
        column_types = self._extract_column_types(parsed_migration)
        
        # Create a query based on the table names and column types
        query = f"JPA entity annotations for tables: {', '.join(table_names)} with column types: {', '.join(column_types)}"
        
        # Get relevant documents
        docs = self.jpa_docs_retriever.get_relevant_documents(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
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
        docs = self.liquibase_docs_retriever.get_relevant_documents(query)
        
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
        # Extract table names from the parsed migration
        table_names = self._extract_table_names(parsed_migration)
        
        # Create a query based on the table names
        query = f"Internal guidelines for JPA entities with tables: {', '.join(table_names)}"
        
        # Get relevant documents
        docs = self.internal_guidelines_retriever.get_relevant_documents(query)
        
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
        # Extract table names from the parsed migration
        table_names = self._extract_table_names(parsed_migration)
        
        # Create a query based on the table names
        query = f"Example migrations for tables: {', '.join(table_names)}"
        
        # Get relevant documents
        docs = self.example_migrations_retriever.get_relevant_documents(query)
        
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
            for changeset in parsed_migration['databaseChangeLog'].get('changeSet', []):
                for change in changeset.get('changes', []):
                    # Each change is a dictionary with a single key (the change type)
                    change_types.update(change.keys())
        
        return list(change_types)
    
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
            for changeset in parsed_migration['databaseChangeLog'].get('changeSet', []):
                for change in changeset.get('changes', []):
                    # Each change is a dictionary with a single key (the change type)
                    for change_type, change_data in change.items():
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
                        elif change_type in ['addPrimaryKey', 'addUniqueConstraint', 'createIndex'] and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
        
        return list(table_names)
    
    def _extract_column_types(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Extract column types from a parsed migration.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of column types.
        """
        column_types = set()
        
        # Extract column types from changesets
        if 'databaseChangeLog' in parsed_migration:
            for changeset in parsed_migration['databaseChangeLog'].get('changeSet', []):
                for change in changeset.get('changes', []):
                    # Each change is a dictionary with a single key (the change type)
                    for change_type, change_data in change.items():
                        # Extract column types based on change type
                        if change_type == 'createTable' and 'columns' in change_data:
                            for column in change_data['columns']:
                                if 'type' in column:
                                    column_types.add(column['type'])
                        elif change_type == 'addColumn' and 'columns' in change_data:
                            for column in change_data['columns']:
                                if 'type' in column:
                                    column_types.add(column['type'])
        
        return list(column_types)
    
    def _combine_context(
        self,
        jpa_docs: List[str],
        liquibase_docs: List[str],
        internal_guidelines: List[str],
        example_migrations: List[str]
    ) -> str:
        """
        Combine context from different sources.
        
        Args:
            jpa_docs: Relevant JPA documentation.
            liquibase_docs: Relevant Liquibase documentation.
            internal_guidelines: Relevant internal guidelines.
            example_migrations: Relevant example migrations.
        
        Returns:
            Combined context.
        """
        context_parts = []
        
        # Add JPA documentation
        if jpa_docs:
            context_parts.append("## JPA Documentation\n\n" + "\n\n".join(jpa_docs))
        
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
        Create a chain for generating entities.
        
        Returns:
            A chain for generating entities.
        """
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a JPA entity generator. Your task is to generate a JPA entity class from a Liquibase migration.
        
        # Migration Content
        ```{format_type}
        {migration_content}
        ```
        
        # Parsed Migration Structure
        ```
        {parsed_migration}
        ```
        
        # Package Name
        {package_name}
        
        # Use Lombok
        {lombok}
        
        # Reference Documentation and Guidelines
        {context}
        
        Please generate a complete and valid JPA entity class based on the migration. Follow these guidelines:
        
        1. Use appropriate JPA annotations (@Entity, @Table, @Column, etc.).
        2. Map database column types to appropriate Java types.
        3. Include appropriate relationships (@OneToMany, @ManyToOne, etc.) based on foreign key constraints.
        4. Use Lombok annotations (@Data, @NoArgsConstructor, etc.) if specified.
        5. Follow Java naming conventions (camelCase for fields, PascalCase for class names).
        6. Include appropriate validation annotations (@NotNull, @Size, etc.) based on column constraints.
        7. Add appropriate comments to explain the purpose of the entity and its fields.
        
        Return ONLY the Java entity class without any additional explanation.
        """)
        
        # Create the chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
