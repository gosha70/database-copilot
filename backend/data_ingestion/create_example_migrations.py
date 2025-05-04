"""
Script to create example Liquibase migrations for testing and as examples.
"""
import os
import logging
import argparse

from backend.config import DOC_CATEGORIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Example XML migration for creating a users table
EXAMPLE_XML_USERS = """<?xml version="1.0" encoding="UTF-8"?>
<databaseChangeLog
    xmlns="http://www.liquibase.org/xml/ns/dbchangelog"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.liquibase.org/xml/ns/dbchangelog
                      http://www.liquibase.org/xml/ns/dbchangelog/dbchangelog-4.20.xsd">

    <!-- Create users table -->
    <changeSet id="1" author="example">
        <comment>Create users table with basic fields</comment>
        <createTable tableName="users">
            <column name="id" type="BIGINT" autoIncrement="true">
                <constraints primaryKey="true" nullable="false"/>
            </column>
            <column name="username" type="VARCHAR(50)">
                <constraints unique="true" nullable="false"/>
            </column>
            <column name="email" type="VARCHAR(100)">
                <constraints unique="true" nullable="false"/>
            </column>
            <column name="password" type="VARCHAR(255)">
                <constraints nullable="false"/>
            </column>
            <column name="created_at" type="TIMESTAMP" defaultValueComputed="CURRENT_TIMESTAMP">
                <constraints nullable="false"/>
            </column>
            <column name="updated_at" type="TIMESTAMP" defaultValueComputed="CURRENT_TIMESTAMP">
                <constraints nullable="false"/>
            </column>
        </createTable>
    </changeSet>
</databaseChangeLog>
"""

# Example XML migration for creating a products table
EXAMPLE_XML_PRODUCTS = """<?xml version="1.0" encoding="UTF-8"?>
<databaseChangeLog
    xmlns="http://www.liquibase.org/xml/ns/dbchangelog"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.liquibase.org/xml/ns/dbchangelog
                      http://www.liquibase.org/xml/ns/dbchangelog/dbchangelog-4.20.xsd">

    <!-- Create products table -->
    <changeSet id="1" author="example">
        <comment>Create products table with basic fields</comment>
        <createTable tableName="products">
            <column name="id" type="BIGINT" autoIncrement="true">
                <constraints primaryKey="true" nullable="false"/>
            </column>
            <column name="name" type="VARCHAR(100)">
                <constraints nullable="false"/>
            </column>
            <column name="description" type="TEXT"/>
            <column name="price" type="DECIMAL(10,2)">
                <constraints nullable="false"/>
            </column>
            <column name="stock" type="INT" defaultValue="0">
                <constraints nullable="false"/>
            </column>
            <column name="created_at" type="TIMESTAMP" defaultValueComputed="CURRENT_TIMESTAMP">
                <constraints nullable="false"/>
            </column>
            <column name="updated_at" type="TIMESTAMP" defaultValueComputed="CURRENT_TIMESTAMP">
                <constraints nullable="false"/>
            </column>
        </createTable>
    </changeSet>
</databaseChangeLog>
"""

# Example XML migration for creating an orders table with foreign keys
EXAMPLE_XML_ORDERS = """<?xml version="1.0" encoding="UTF-8"?>
<databaseChangeLog
    xmlns="http://www.liquibase.org/xml/ns/dbchangelog"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.liquibase.org/xml/ns/dbchangelog
                      http://www.liquibase.org/xml/ns/dbchangelog/dbchangelog-4.20.xsd">

    <!-- Create orders table with foreign keys -->
    <changeSet id="1" author="example">
        <comment>Create orders table with references to users</comment>
        <createTable tableName="orders">
            <column name="id" type="BIGINT" autoIncrement="true">
                <constraints primaryKey="true" nullable="false"/>
            </column>
            <column name="user_id" type="BIGINT">
                <constraints nullable="false"/>
            </column>
            <column name="status" type="VARCHAR(20)" defaultValue="PENDING">
                <constraints nullable="false"/>
            </column>
            <column name="total_amount" type="DECIMAL(10,2)">
                <constraints nullable="false"/>
            </column>
            <column name="created_at" type="TIMESTAMP" defaultValueComputed="CURRENT_TIMESTAMP">
                <constraints nullable="false"/>
            </column>
            <column name="updated_at" type="TIMESTAMP" defaultValueComputed="CURRENT_TIMESTAMP">
                <constraints nullable="false"/>
            </column>
        </createTable>
        
        <addForeignKeyConstraint 
            baseTableName="orders" 
            baseColumnNames="user_id" 
            constraintName="fk_orders_users" 
            referencedTableName="users" 
            referencedColumnNames="id"
            onDelete="CASCADE"
            onUpdate="RESTRICT"/>
    </changeSet>
</databaseChangeLog>
"""

# Example XML migration for creating an order_items table with foreign keys
EXAMPLE_XML_ORDER_ITEMS = """<?xml version="1.0" encoding="UTF-8"?>
<databaseChangeLog
    xmlns="http://www.liquibase.org/xml/ns/dbchangelog"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.liquibase.org/xml/ns/dbchangelog
                      http://www.liquibase.org/xml/ns/dbchangelog/dbchangelog-4.20.xsd">

    <!-- Create order_items table with foreign keys -->
    <changeSet id="1" author="example">
        <comment>Create order_items table with references to orders and products</comment>
        <createTable tableName="order_items">
            <column name="id" type="BIGINT" autoIncrement="true">
                <constraints primaryKey="true" nullable="false"/>
            </column>
            <column name="order_id" type="BIGINT">
                <constraints nullable="false"/>
            </column>
            <column name="product_id" type="BIGINT">
                <constraints nullable="false"/>
            </column>
            <column name="quantity" type="INT" defaultValue="1">
                <constraints nullable="false"/>
            </column>
            <column name="price" type="DECIMAL(10,2)">
                <constraints nullable="false"/>
            </column>
            <column name="created_at" type="TIMESTAMP" defaultValueComputed="CURRENT_TIMESTAMP">
                <constraints nullable="false"/>
            </column>
            <column name="updated_at" type="TIMESTAMP" defaultValueComputed="CURRENT_TIMESTAMP">
                <constraints nullable="false"/>
            </column>
        </createTable>
        
        <addForeignKeyConstraint 
            baseTableName="order_items" 
            baseColumnNames="order_id" 
            constraintName="fk_order_items_orders" 
            referencedTableName="orders" 
            referencedColumnNames="id"
            onDelete="CASCADE"
            onUpdate="RESTRICT"/>
            
        <addForeignKeyConstraint 
            baseTableName="order_items" 
            baseColumnNames="product_id" 
            constraintName="fk_order_items_products" 
            referencedTableName="products" 
            referencedColumnNames="id"
            onDelete="RESTRICT"
            onUpdate="RESTRICT"/>
    </changeSet>
</databaseChangeLog>
"""

# Example XML migration for adding columns to an existing table
EXAMPLE_XML_ADD_COLUMNS = """<?xml version="1.0" encoding="UTF-8"?>
<databaseChangeLog
    xmlns="http://www.liquibase.org/xml/ns/dbchangelog"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.liquibase.org/xml/ns/dbchangelog
                      http://www.liquibase.org/xml/ns/dbchangelog/dbchangelog-4.20.xsd">

    <!-- Add columns to users table -->
    <changeSet id="1" author="example">
        <comment>Add additional columns to users table</comment>
        <addColumn tableName="users">
            <column name="first_name" type="VARCHAR(50)"/>
            <column name="last_name" type="VARCHAR(50)"/>
            <column name="phone" type="VARCHAR(20)"/>
            <column name="is_active" type="BOOLEAN" defaultValueBoolean="true">
                <constraints nullable="false"/>
            </column>
        </addColumn>
    </changeSet>
</databaseChangeLog>
"""

# Example XML migration for creating indexes
EXAMPLE_XML_INDEXES = """<?xml version="1.0" encoding="UTF-8"?>
<databaseChangeLog
    xmlns="http://www.liquibase.org/xml/ns/dbchangelog"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.liquibase.org/xml/ns/dbchangelog
                      http://www.liquibase.org/xml/ns/dbchangelog/dbchangelog-4.20.xsd">

    <!-- Create indexes for better performance -->
    <changeSet id="1" author="example">
        <comment>Create indexes for better query performance</comment>
        
        <createIndex indexName="idx_users_email" tableName="users">
            <column name="email"/>
        </createIndex>
        
        <createIndex indexName="idx_products_name" tableName="products">
            <column name="name"/>
        </createIndex>
        
        <createIndex indexName="idx_orders_user_id" tableName="orders">
            <column name="user_id"/>
        </createIndex>
        
        <createIndex indexName="idx_orders_status" tableName="orders">
            <column name="status"/>
        </createIndex>
        
        <createIndex indexName="idx_order_items_order_id" tableName="order_items">
            <column name="order_id"/>
        </createIndex>
        
        <createIndex indexName="idx_order_items_product_id" tableName="order_items">
            <column name="product_id"/>
        </createIndex>
    </changeSet>
</databaseChangeLog>
"""

# Example YAML migration for creating a users table
EXAMPLE_YAML_USERS = """databaseChangeLog:
  - changeSet:
      id: 1
      author: example
      comment: Create users table with basic fields
      changes:
        - createTable:
            tableName: users
            columns:
              - column:
                  name: id
                  type: BIGINT
                  autoIncrement: true
                  constraints:
                    primaryKey: true
                    nullable: false
              - column:
                  name: username
                  type: VARCHAR(50)
                  constraints:
                    unique: true
                    nullable: false
              - column:
                  name: email
                  type: VARCHAR(100)
                  constraints:
                    unique: true
                    nullable: false
              - column:
                  name: password
                  type: VARCHAR(255)
                  constraints:
                    nullable: false
              - column:
                  name: created_at
                  type: TIMESTAMP
                  defaultValueComputed: CURRENT_TIMESTAMP
                  constraints:
                    nullable: false
              - column:
                  name: updated_at
                  type: TIMESTAMP
                  defaultValueComputed: CURRENT_TIMESTAMP
                  constraints:
                    nullable: false
"""

# Example YAML migration for creating a products table
EXAMPLE_YAML_PRODUCTS = """databaseChangeLog:
  - changeSet:
      id: 1
      author: example
      comment: Create products table with basic fields
      changes:
        - createTable:
            tableName: products
            columns:
              - column:
                  name: id
                  type: BIGINT
                  autoIncrement: true
                  constraints:
                    primaryKey: true
                    nullable: false
              - column:
                  name: name
                  type: VARCHAR(100)
                  constraints:
                    nullable: false
              - column:
                  name: description
                  type: TEXT
              - column:
                  name: price
                  type: DECIMAL(10,2)
                  constraints:
                    nullable: false
              - column:
                  name: stock
                  type: INT
                  defaultValue: 0
                  constraints:
                    nullable: false
              - column:
                  name: created_at
                  type: TIMESTAMP
                  defaultValueComputed: CURRENT_TIMESTAMP
                  constraints:
                    nullable: false
              - column:
                  name: updated_at
                  type: TIMESTAMP
                  defaultValueComputed: CURRENT_TIMESTAMP
                  constraints:
                    nullable: false
"""

# Example YAML migration for creating an orders table with foreign keys
EXAMPLE_YAML_ORDERS = """databaseChangeLog:
  - changeSet:
      id: 1
      author: example
      comment: Create orders table with references to users
      changes:
        - createTable:
            tableName: orders
            columns:
              - column:
                  name: id
                  type: BIGINT
                  autoIncrement: true
                  constraints:
                    primaryKey: true
                    nullable: false
              - column:
                  name: user_id
                  type: BIGINT
                  constraints:
                    nullable: false
              - column:
                  name: status
                  type: VARCHAR(20)
                  defaultValue: PENDING
                  constraints:
                    nullable: false
              - column:
                  name: total_amount
                  type: DECIMAL(10,2)
                  constraints:
                    nullable: false
              - column:
                  name: created_at
                  type: TIMESTAMP
                  defaultValueComputed: CURRENT_TIMESTAMP
                  constraints:
                    nullable: false
              - column:
                  name: updated_at
                  type: TIMESTAMP
                  defaultValueComputed: CURRENT_TIMESTAMP
                  constraints:
                    nullable: false
        - addForeignKeyConstraint:
            baseTableName: orders
            baseColumnNames: user_id
            constraintName: fk_orders_users
            referencedTableName: users
            referencedColumnNames: id
            onDelete: CASCADE
            onUpdate: RESTRICT
"""

def save_example_to_file(content: str, file_path: str) -> None:
    """
    Save example content to a file.
    
    Args:
        content: Content to save.
        file_path: Path to save the file to.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved example to {file_path}")
    except Exception as e:
        logger.error(f"Error saving example to {file_path}: {e}")

def create_example_migrations(output_dir: str) -> None:
    """
    Create example Liquibase migrations.
    
    Args:
        output_dir: Directory to save the examples to.
    """
    logger.info(f"Creating example Liquibase migrations in {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create XML examples directory
    xml_dir = os.path.join(output_dir, "xml")
    os.makedirs(xml_dir, exist_ok=True)
    
    # Create YAML examples directory
    yaml_dir = os.path.join(output_dir, "yaml")
    os.makedirs(yaml_dir, exist_ok=True)
    
    # Save XML examples
    save_example_to_file(EXAMPLE_XML_USERS, os.path.join(xml_dir, "users.xml"))
    save_example_to_file(EXAMPLE_XML_PRODUCTS, os.path.join(xml_dir, "products.xml"))
    save_example_to_file(EXAMPLE_XML_ORDERS, os.path.join(xml_dir, "orders.xml"))
    save_example_to_file(EXAMPLE_XML_ORDER_ITEMS, os.path.join(xml_dir, "order_items.xml"))
    save_example_to_file(EXAMPLE_XML_ADD_COLUMNS, os.path.join(xml_dir, "add_columns.xml"))
    save_example_to_file(EXAMPLE_XML_INDEXES, os.path.join(xml_dir, "indexes.xml"))
    
    # Save YAML examples
    save_example_to_file(EXAMPLE_YAML_USERS, os.path.join(yaml_dir, "users.yaml"))
    save_example_to_file(EXAMPLE_YAML_PRODUCTS, os.path.join(yaml_dir, "products.yaml"))
    save_example_to_file(EXAMPLE_YAML_ORDERS, os.path.join(yaml_dir, "orders.yaml"))
    
    logger.info(f"Finished creating example Liquibase migrations in {output_dir}")

def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(description="Create example Liquibase migrations")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DOC_CATEGORIES["example_migrations"],
        help="Directory to save the examples to"
    )
    
    args = parser.parse_args()
    
    create_example_migrations(args.output_dir)

if __name__ == "__main__":
    main()
