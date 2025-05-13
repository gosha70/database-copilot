# Java Files Support for Database Copilot

This directory contains Java entity classes that can be used as reference documentation in the Database Copilot application. These files are indexed in the vector store and can be queried through the Q/A System.

## Available Java Files

- `Customer.java` - Customer entity class with JPA annotations
- `Order.java` - Order entity class with JPA annotations
- `OrderItem.java` - OrderItem entity class with JPA annotations
- `Product.java` - Product entity class with JPA annotations

## How to Use

1. **Add Java Files**: Place your Java entity classes in this directory.

2. **Build Vector Store**: Run the following command to build the vector store for Java files:

   ```bash
   python build_vector_store.py --category java_files
   ```

3. **Query Java Files**: In the Database Copilot application, go to the "Q/A System" tab and select "java" from the "Documentation Category" dropdown to specifically query Java files.

## Benefits

- **Entity Design Reference**: Use these Java entity classes as reference when designing new entities.
- **JPA Annotation Examples**: See examples of JPA annotations for various use cases.
- **Relationship Mapping**: Learn how to map relationships between entities (OneToMany, ManyToOne, etc.).
- **Best Practices**: Follow best practices for entity design and implementation.

## Integration with Database Migrations

The Java entity classes in this directory can be used in conjunction with Liquibase migrations to:

1. Generate Liquibase migrations from existing Java entities
2. Generate Java entities from existing Liquibase migrations
3. Ensure consistency between database schema and Java entities

## Adding Your Own Java Files

When adding your own Java files to this directory, consider the following:

1. Include comprehensive JavaDoc comments to provide context
2. Use proper JPA annotations to define the entity structure
3. Implement appropriate relationships between entities
4. Include business logic methods that demonstrate entity behavior

After adding new Java files, rebuild the vector store to include them in the search index.
