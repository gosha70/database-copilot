"""
Liquibase migration parser for XML and YAML formats.
"""
import os
import logging
import xml.etree.ElementTree as ET
import yaml
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class LiquibaseParser:
    """
    Parser for Liquibase migration files in XML and YAML formats.
    """
    
    def __init__(self):
        """
        Initialize the parser.
        """
        self.namespaces = {
            'lb': 'http://www.liquibase.org/xml/ns/dbchangelog',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Liquibase migration file.
        
        Args:
            file_path: Path to the Liquibase migration file.
        
        Returns:
            A dictionary containing the parsed migration.
        """
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return {}
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.xml']:
            return self.parse_xml_file(file_path)
        elif file_ext in ['.yaml', '.yml']:
            return self.parse_yaml_file(file_path)
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            return {}
    
    def parse_xml_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Liquibase XML migration file.
        
        Args:
            file_path: Path to the Liquibase XML migration file.
        
        Returns:
            A dictionary containing the parsed migration.
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Handle namespace if present
            if '}' in root.tag:
                ns = root.tag.split('}')[0] + '}'
                self.namespaces['lb'] = ns[1:-1]  # Remove { and }
            
            # Parse databaseChangeLog
            if root.tag.endswith('databaseChangeLog'):
                return self._parse_database_change_log(root)
            else:
                logger.error(f"Invalid Liquibase XML file: {file_path}")
                return {}
        
        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {e}")
            return {}
    
    def parse_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Liquibase YAML migration file.
        
        Args:
            file_path: Path to the Liquibase YAML migration file.
        
        Returns:
            A dictionary containing the parsed migration.
        """
        try:
            # First try to preprocess the YAML file to handle Liquibase-specific syntax
            with open(file_path, 'r') as f:
                yaml_content_str = f.read()
            
            # Preprocess the YAML content to handle Liquibase-specific syntax
            preprocessed_yaml = self._preprocess_yaml(yaml_content_str)
            
            # Parse the preprocessed YAML
            yaml_content = yaml.safe_load(preprocessed_yaml)
            
            if not yaml_content:
                logger.error(f"Empty YAML file: {file_path}")
                return {}
            
            # Check if it's a valid Liquibase YAML file
            if 'databaseChangeLog' not in yaml_content:
                logger.error(f"Invalid Liquibase YAML file: {file_path}")
                return {}
            
            return yaml_content
        
        except Exception as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            # Return a dictionary with an error message
            return {"error": str(e)}
    
    def _preprocess_yaml(self, yaml_content: str) -> str:
        """
        Preprocess YAML content to handle Liquibase-specific syntax.
        
        Args:
            yaml_content: The YAML content to preprocess.
        
        Returns:
            The preprocessed YAML content.
        """
        # Handle Liquibase-specific syntax issues
        
        # Common issues with Liquibase YAML:
        # 1. Unquoted values with special characters
        # 2. Inconsistent indentation
        # 3. Special syntax for preConditions with "not:" operator
        
        # Split the content into lines
        lines = yaml_content.split('\n')
        processed_lines = []
        in_precondition_block = False
        
        for i, line in enumerate(lines):
            # Check for preCondition blocks with "not:" operator
            if "preConditions:" in line:
                in_precondition_block = True
            
            # Handle special syntax in preCondition blocks
            if in_precondition_block and "- not:" in line:
                # This is a common issue in Liquibase YAML that causes parsing errors
                # Convert "- not:" to a valid YAML format
                indentation = line.index("- not:")
                processed_line = line.replace("- not:", "- notCondition:")
                processed_lines.append(processed_line)
                continue
            
            # Check if we're exiting a preCondition block
            if in_precondition_block and line.strip() and not line.lstrip().startswith("-"):
                # Check indentation level to see if we've exited the block
                if i > 0 and lines[i-1].startswith(" ") and not line.startswith(" "):
                    in_precondition_block = False
            
            # Add the line as is
            processed_lines.append(line)
        
        # Join the processed lines back into a string
        return '\n'.join(processed_lines)
    
    def _parse_database_change_log(self, root: ET.Element) -> Dict[str, Any]:
        """
        Parse a databaseChangeLog element.
        
        Args:
            root: The root element of the XML tree.
        
        Returns:
            A dictionary containing the parsed databaseChangeLog.
        """
        result = {
            'databaseChangeLog': {
                'changeSet': []
            }
        }
        
        # Parse changeSet elements
        for changeset in root.findall('.//lb:changeSet', self.namespaces):
            changeset_data = self._parse_change_set(changeset)
            if changeset_data:
                result['databaseChangeLog']['changeSet'].append(changeset_data)
        
        return result
    
    def _parse_change_set(self, changeset: ET.Element) -> Dict[str, Any]:
        """
        Parse a changeSet element.
        
        Args:
            changeset: The changeSet element.
        
        Returns:
            A dictionary containing the parsed changeSet.
        """
        changeset_data = {
            'id': changeset.get('id', ''),
            'author': changeset.get('author', ''),
            'changes': []
        }
        
        # Add other attributes
        for attr, value in changeset.attrib.items():
            if attr not in ['id', 'author']:
                changeset_data[attr] = value
        
        # Parse change elements
        for change in changeset:
            # Skip non-element nodes
            if not isinstance(change.tag, str):
                continue
            
            # Get the change type (tag name without namespace)
            if '}' in change.tag:
                change_type = change.tag.split('}')[1]
            else:
                change_type = change.tag
            
            # Parse specific change types
            if change_type == 'createTable':
                change_data = self._parse_create_table(change)
            elif change_type == 'addColumn':
                change_data = self._parse_add_column(change)
            elif change_type == 'dropTable':
                change_data = self._parse_drop_table(change)
            elif change_type == 'addForeignKeyConstraint':
                change_data = self._parse_add_foreign_key_constraint(change)
            elif change_type == 'addPrimaryKey':
                change_data = self._parse_add_primary_key(change)
            elif change_type == 'addUniqueConstraint':
                change_data = self._parse_add_unique_constraint(change)
            elif change_type == 'createIndex':
                change_data = self._parse_create_index(change)
            elif change_type == 'sql':
                change_data = self._parse_sql(change)
            else:
                # Generic parsing for other change types
                change_data = self._parse_generic_change(change, change_type)
            
            if change_data:
                changeset_data['changes'].append(change_data)
        
        return changeset_data
    
    def _parse_create_table(self, element: ET.Element) -> Dict[str, Any]:
        """
        Parse a createTable element.
        
        Args:
            element: The createTable element.
        
        Returns:
            A dictionary containing the parsed createTable.
        """
        table_data = {
            'createTable': {
                'tableName': element.get('tableName', ''),
                'columns': []
            }
        }
        
        # Add other attributes
        for attr, value in element.attrib.items():
            if attr != 'tableName':
                table_data['createTable'][attr] = value
        
        # Parse column elements
        for column in element.findall('.//lb:column', self.namespaces):
            column_data = {
                'name': column.get('name', ''),
                'type': column.get('type', '')
            }
            
            # Add other attributes
            for attr, value in column.attrib.items():
                if attr not in ['name', 'type']:
                    column_data[attr] = value
            
            # Parse constraints
            constraints = column.find('.//lb:constraints', self.namespaces)
            if constraints is not None:
                column_data['constraints'] = {}
                for attr, value in constraints.attrib.items():
                    column_data['constraints'][attr] = value
            
            table_data['createTable']['columns'].append(column_data)
        
        return table_data
    
    def _parse_add_column(self, element: ET.Element) -> Dict[str, Any]:
        """
        Parse an addColumn element.
        
        Args:
            element: The addColumn element.
        
        Returns:
            A dictionary containing the parsed addColumn.
        """
        column_data = {
            'addColumn': {
                'tableName': element.get('tableName', ''),
                'columns': []
            }
        }
        
        # Add other attributes
        for attr, value in element.attrib.items():
            if attr != 'tableName':
                column_data['addColumn'][attr] = value
        
        # Parse column elements
        for column in element.findall('.//lb:column', self.namespaces):
            col_data = {
                'name': column.get('name', ''),
                'type': column.get('type', '')
            }
            
            # Add other attributes
            for attr, value in column.attrib.items():
                if attr not in ['name', 'type']:
                    col_data[attr] = value
            
            # Parse constraints
            constraints = column.find('.//lb:constraints', self.namespaces)
            if constraints is not None:
                col_data['constraints'] = {}
                for attr, value in constraints.attrib.items():
                    col_data['constraints'][attr] = value
            
            column_data['addColumn']['columns'].append(col_data)
        
        return column_data
    
    def _parse_drop_table(self, element: ET.Element) -> Dict[str, Any]:
        """
        Parse a dropTable element.
        
        Args:
            element: The dropTable element.
        
        Returns:
            A dictionary containing the parsed dropTable.
        """
        drop_data = {
            'dropTable': {
                'tableName': element.get('tableName', '')
            }
        }
        
        # Add other attributes
        for attr, value in element.attrib.items():
            if attr != 'tableName':
                drop_data['dropTable'][attr] = value
        
        return drop_data
    
    def _parse_add_foreign_key_constraint(self, element: ET.Element) -> Dict[str, Any]:
        """
        Parse an addForeignKeyConstraint element.
        
        Args:
            element: The addForeignKeyConstraint element.
        
        Returns:
            A dictionary containing the parsed addForeignKeyConstraint.
        """
        fk_data = {
            'addForeignKeyConstraint': {
                'constraintName': element.get('constraintName', ''),
                'baseTableName': element.get('baseTableName', ''),
                'baseColumnNames': element.get('baseColumnNames', ''),
                'referencedTableName': element.get('referencedTableName', ''),
                'referencedColumnNames': element.get('referencedColumnNames', '')
            }
        }
        
        # Add other attributes
        for attr, value in element.attrib.items():
            if attr not in ['constraintName', 'baseTableName', 'baseColumnNames', 'referencedTableName', 'referencedColumnNames']:
                fk_data['addForeignKeyConstraint'][attr] = value
        
        return fk_data
    
    def _parse_add_primary_key(self, element: ET.Element) -> Dict[str, Any]:
        """
        Parse an addPrimaryKey element.
        
        Args:
            element: The addPrimaryKey element.
        
        Returns:
            A dictionary containing the parsed addPrimaryKey.
        """
        pk_data = {
            'addPrimaryKey': {
                'constraintName': element.get('constraintName', ''),
                'tableName': element.get('tableName', ''),
                'columnNames': element.get('columnNames', '')
            }
        }
        
        # Add other attributes
        for attr, value in element.attrib.items():
            if attr not in ['constraintName', 'tableName', 'columnNames']:
                pk_data['addPrimaryKey'][attr] = value
        
        return pk_data
    
    def _parse_add_unique_constraint(self, element: ET.Element) -> Dict[str, Any]:
        """
        Parse an addUniqueConstraint element.
        
        Args:
            element: The addUniqueConstraint element.
        
        Returns:
            A dictionary containing the parsed addUniqueConstraint.
        """
        uc_data = {
            'addUniqueConstraint': {
                'constraintName': element.get('constraintName', ''),
                'tableName': element.get('tableName', ''),
                'columnNames': element.get('columnNames', '')
            }
        }
        
        # Add other attributes
        for attr, value in element.attrib.items():
            if attr not in ['constraintName', 'tableName', 'columnNames']:
                uc_data['addUniqueConstraint'][attr] = value
        
        return uc_data
    
    def _parse_create_index(self, element: ET.Element) -> Dict[str, Any]:
        """
        Parse a createIndex element.
        
        Args:
            element: The createIndex element.
        
        Returns:
            A dictionary containing the parsed createIndex.
        """
        index_data = {
            'createIndex': {
                'indexName': element.get('indexName', ''),
                'tableName': element.get('tableName', ''),
                'columns': []
            }
        }
        
        # Add other attributes
        for attr, value in element.attrib.items():
            if attr not in ['indexName', 'tableName']:
                index_data['createIndex'][attr] = value
        
        # Parse column elements
        for column in element.findall('.//lb:column', self.namespaces):
            col_data = {
                'name': column.get('name', '')
            }
            
            # Add other attributes
            for attr, value in column.attrib.items():
                if attr != 'name':
                    col_data[attr] = value
            
            index_data['createIndex']['columns'].append(col_data)
        
        return index_data
    
    def _parse_sql(self, element: ET.Element) -> Dict[str, Any]:
        """
        Parse an sql element.
        
        Args:
            element: The sql element.
        
        Returns:
            A dictionary containing the parsed sql.
        """
        sql_data = {
            'sql': {
                'sql': element.text.strip() if element.text else ''
            }
        }
        
        # Add attributes
        for attr, value in element.attrib.items():
            sql_data['sql'][attr] = value
        
        return sql_data
    
    def _parse_generic_change(self, element: ET.Element, change_type: str) -> Dict[str, Any]:
        """
        Parse a generic change element.
        
        Args:
            element: The change element.
            change_type: The type of change.
        
        Returns:
            A dictionary containing the parsed change.
        """
        change_data = {
            change_type: {}
        }
        
        # Add attributes
        for attr, value in element.attrib.items():
            change_data[change_type][attr] = value
        
        # Add child elements
        for child in element:
            # Skip non-element nodes
            if not isinstance(child.tag, str):
                continue
            
            # Get the child type (tag name without namespace)
            if '}' in child.tag:
                child_type = child.tag.split('}')[1]
            else:
                child_type = child.tag
            
            # Add child element
            if child_type not in change_data[change_type]:
                change_data[change_type][child_type] = []
            
            child_data = {}
            
            # Add attributes
            for attr, value in child.attrib.items():
                child_data[attr] = value
            
            # Add text content if present
            if child.text and child.text.strip():
                child_data['_text'] = child.text.strip()
            
            change_data[change_type][child_type].append(child_data)
        
        return change_data
