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
        logger.info(f"Parsed migration structure: {parsed_migration}")
        
        # Extract detailed table structure
        table_structure = self._extract_table_structure(parsed_migration)
        if not table_structure or not table_structure['table_name']:
            logger.error("Failed to extract table structure from migration")
            return "Error: Could not extract table structure from the migration file."
        
        logger.info(f"Extracted table structure: {table_structure}")
        
        # Generate a basic entity template from the table structure
        entity_template = self._generate_entity_template(table_structure, package_name, lombok)
        logger.info(f"Generated entity template: {entity_template[:200]}...")
        
        # Get relevant documents from different categories
        jpa_docs = self._get_relevant_jpa_docs(parsed_migration)
        liquibase_docs = self._get_relevant_liquibase_docs(parsed_migration)
        internal_guidelines = self._get_relevant_internal_guidelines(parsed_migration)
        example_migrations = self._get_relevant_example_migrations(parsed_migration)
        
        # Combine all relevant documents
        context = self._combine_context(jpa_docs, liquibase_docs, internal_guidelines, example_migrations)
        
        # Create the generation chain
        generation_chain = self._create_generation_chain()
        
        # Prepare input for the LLM
        input_data = {
            "entity_template": entity_template,
            "table_structure": str(table_structure),
            "package_name": package_name,
            "lombok": lombok,
            "context": context
        }
        
        # Log the input data for debugging
        logger.info(f"Input data for LLM - package_name: {package_name}, lombok: {lombok}")
        logger.info(f"Table structure for LLM: {str(table_structure)}")
        
        # Generate the entity
        entity = generation_chain.invoke(input_data)
        
        # Log a preview of the generated entity
        logger.info(f"Generated entity preview: {entity[:200]}...")
        
        # Ensure the entity has the correct table name
        if table_structure['table_name'] not in entity:
            logger.warning(f"Generated entity does not contain the correct table name: {table_structure['table_name']}")
            # Fall back to the template if the LLM generated something incorrect
            return entity_template
        
        return entity
        
    def _generate_entity_template(self, table_structure: Dict[str, Any], package_name: str, lombok: bool) -> str:
        """
        Generate a basic entity template from the table structure.
        
        Args:
            table_structure: The extracted table structure.
            package_name: The package name for the entity.
            lombok: Whether to use Lombok annotations.
            
        Returns:
            A basic entity template.
        """
        table_name = table_structure['table_name']
        # Convert snake_case to PascalCase for class name
        class_name = ''.join(word.capitalize() for word in table_name.split('_'))
        
        # Start building the entity
        entity_lines = []
        
        # Package and imports
        entity_lines.append(f"package {package_name};")
        entity_lines.append("")
        entity_lines.append("import javax.persistence.*;")
        
        # Add additional imports
        if any(col.get('type') == '${uuidType}' for col in table_structure['columns']):
            entity_lines.append("import java.util.UUID;")
        
        # Add Lombok imports if needed
        if lombok:
            entity_lines.append("import lombok.Data;")
            entity_lines.append("import lombok.NoArgsConstructor;")
            entity_lines.append("import lombok.AllArgsConstructor;")
        
        entity_lines.append("")
        
        # Add Lombok annotations if needed
        if lombok:
            entity_lines.append("@Data")
            entity_lines.append("@NoArgsConstructor")
            entity_lines.append("@AllArgsConstructor")
        
        # Add JPA annotations
        entity_lines.append("@Entity")
        entity_lines.append(f"@Table(name = \"{table_name}\")")
        entity_lines.append(f"public class {class_name} {{")
        
        # Add fields for each column
        for column in table_structure['columns']:
            entity_lines.append("")
            
            # Add column annotations
            if column.get('primary_key'):
                entity_lines.append("    @Id")
                if column.get('auto_increment'):
                    entity_lines.append("    @GeneratedValue(strategy = GenerationType.IDENTITY)")
            
            nullable = "false" if not column.get('nullable', True) else "true"
            unique = "true" if column.get('unique', False) else "false"
            
            entity_lines.append(f"    @Column(name = \"{column.get('name')}\", nullable = {nullable}, unique = {unique})")
            
            # Map column type to Java type
            java_type = self._map_db_type_to_java(column.get('type'))
            
            # Add field
            field_name = self._to_camel_case(column.get('name'))
            entity_lines.append(f"    private {java_type} {field_name};")
        
        # Add foreign key relationships
        for fk in table_structure['foreign_keys']:
            entity_lines.append("")
            referenced_table = fk.get('referenced_table')
            referenced_class = ''.join(word.capitalize() for word in referenced_table.split('_'))
            field_name = self._to_camel_case(referenced_table)
            
            entity_lines.append(f"    @ManyToOne")
            entity_lines.append(f"    @JoinColumn(name = \"{fk.get('base_column')}\", referencedColumnName = \"{fk.get('referenced_column')}\")")
            entity_lines.append(f"    private {referenced_class} {field_name};")
        
        # Close the class
        entity_lines.append("}")
        
        return "\n".join(entity_lines)
    
    def _map_db_type_to_java(self, db_type: str) -> str:
        """
        Map database column type to Java type.
        
        Args:
            db_type: The database column type.
            
        Returns:
            The corresponding Java type.
        """
        if not db_type:
            return "String"
            
        type_mapping = {
            "${longType}": "Long",
            "${uuidType}": "String",  # or UUID
            "${largeStringType}": "String",
            "VARCHAR": "String",
            "BIGINT": "Long",
            "INTEGER": "Integer",
            "INT": "Integer",
            "BOOLEAN": "Boolean",
            "TIMESTAMP": "java.time.LocalDateTime",
            "DATE": "java.time.LocalDate",
            "DECIMAL": "java.math.BigDecimal",
            "DOUBLE": "Double",
            "FLOAT": "Float"
        }
        
        # Try to find an exact match
        if db_type in type_mapping:
            return type_mapping[db_type]
        
        # Try to find a partial match
        for key, value in type_mapping.items():
            if key in db_type:
                return value
        
        # Default to String for unknown types
        return "String"
    
    def _to_camel_case(self, snake_case: str) -> str:
        """
        Convert snake_case to camelCase.
        
        Args:
            snake_case: The snake_case string.
            
        Returns:
            The camelCase string.
        """
        if not snake_case:
            return ""
            
        # Split by underscore and capitalize each word except the first
        words = snake_case.split('_')
        return words[0] + ''.join(word.capitalize() for word in words[1:])
        
    def _extract_table_structure(self, parsed_migration: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract detailed table structure from a parsed migration.
        
        Args:
            parsed_migration: The parsed migration.
            
        Returns:
            A dictionary containing the table structure details.
        """
        table_structure = {
            "table_name": None,
            "columns": [],
            "primary_key": None,
            "foreign_keys": [],
            "unique_constraints": []
        }
        
        # Extract table structure from changesets
        if 'databaseChangeLog' not in parsed_migration:
            return None
            
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
            
            # Extract table structure from each change
            for change in changes_list:
                if not isinstance(change, dict):
                    continue
                
                for change_type, change_data in change.items():
                    if not isinstance(change_data, dict):
                        continue
                        
                    # Extract table name and columns from createTable
                    if change_type == 'createTable' and 'tableName' in change_data:
                        table_structure['table_name'] = change_data['tableName']
                        
                        if 'columns' in change_data:
                            for column_data in change_data['columns']:
                                if isinstance(column_data, dict) and 'column' in column_data:
                                    column = column_data['column']
                                    column_info = {
                                        "name": column.get('name'),
                                        "type": column.get('type'),
                                        "nullable": True,
                                        "primary_key": False,
                                        "unique": False,
                                        "auto_increment": column.get('autoIncrement', False)
                                    }
                                    
                                    # Extract constraints
                                    if 'constraints' in column:
                                        constraints = column['constraints']
                                        if constraints.get('primaryKey'):
                                            column_info['primary_key'] = True
                                            table_structure['primary_key'] = column.get('name')
                                        if constraints.get('nullable') is False:
                                            column_info['nullable'] = False
                                        if constraints.get('unique'):
                                            column_info['unique'] = True
                                            table_structure['unique_constraints'].append({
                                                "column_name": column.get('name'),
                                                "constraint_name": constraints.get('uniqueConstraintName', '')
                                            })
                                    
                                    table_structure['columns'].append(column_info)
                    
                    # Extract foreign key constraints
                    elif change_type == 'addForeignKeyConstraint':
                        if all(k in change_data for k in ['constraintName', 'baseTableName', 'baseColumnNames', 'referencedTableName', 'referencedColumnNames']):
                            fk_info = {
                                "constraint_name": change_data['constraintName'],
                                "base_table": change_data['baseTableName'],
                                "base_column": change_data['baseColumnNames'],
                                "referenced_table": change_data['referencedTableName'],
                                "referenced_column": change_data['referencedColumnNames'],
                                "on_delete": change_data.get('onDelete', '')
                            }
                            table_structure['foreign_keys'].append(fk_info)
        
        return table_structure
    
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
                
                # Extract column types from each change
                for change in changes_list:
                    if not isinstance(change, dict):
                        continue
                    
                    for change_type, change_data in change.items():
                        if not isinstance(change_data, dict):
                            continue
                            
                        # Extract column types based on change type
                        if change_type == 'createTable' and 'columns' in change_data:
                            for column in change_data['columns']:
                                if isinstance(column, dict) and 'type' in column:
                                    column_types.add(column['type'])
                        elif change_type == 'addColumn' and 'columns' in change_data:
                            for column in change_data['columns']:
                                if isinstance(column, dict) and 'type' in column:
                                    column_types.add(column['type'])
        
        return list(column_types) if column_types else ["unknown"]
    
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
        You are a JPA entity generator. Your task is to review and enhance the provided entity template.

        # Entity Template (BASE YOUR RESPONSE ON THIS)
        ```java
        {entity_template}
        ```
        
        # Extracted Table Structure
        ```
        {table_structure}
        ```
        
        # Package Name
        {package_name}
        
        # Use Lombok
        {lombok}
        
        # Reference Documentation and Guidelines
        {context}
        
        IMPORTANT INSTRUCTIONS:
        1. Start with the entity template above and enhance it. DO NOT create a completely different entity.
        2. Keep the same table name, class name, and all fields from the template.
        3. You may add JavaDoc comments, improve annotations, or add validation.
        4. You may improve type mappings if needed:
           - ${{longType}} -> Long
           - ${{uuidType}} -> String or UUID
           - ${{largeStringType}} -> String
        5. Keep all relationships from the template.
        6. Use Lombok annotations if lombok is true.
        7. Follow Java naming conventions (camelCase for fields, PascalCase for class names).
        
        Return ONLY the complete Java entity class without any additional explanation or placeholders.
        """)
        
        # Create the chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
