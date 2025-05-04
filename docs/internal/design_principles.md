# Database Design Principles

## Normalization
- Follow at least Third Normal Form (3NF) for most tables
- Denormalize only when there's a clear performance benefit and it's documented

## Data Types
- Use the most appropriate data type for each column
- Use `VARCHAR` instead of `CHAR` unless the length is fixed
- Use `TEXT` for long text fields
- Use `TIMESTAMP` for date/time fields that need timezone information
- Use `DECIMAL` for currency and precise numeric values, not `FLOAT` or `DOUBLE`
- Use `BIGINT` for IDs and other numeric identifiers to avoid future migration issues

## Constraints
- Every table must have a primary key
- Use foreign key constraints to enforce referential integrity
- Use `NOT NULL` constraint for columns that should not be null
- Use unique constraints for columns that should have unique values
- Use check constraints to enforce business rules

## Indexes
- Add indexes to columns used frequently in WHERE clauses
- Add indexes to columns used in JOIN conditions
- Add indexes to columns used in ORDER BY clauses
- Consider adding indexes to columns with high cardinality
- Avoid over-indexing as it can slow down write operations

## Performance
- Limit the use of triggers and stored procedures
- Avoid using ORM-generated schemas without review
- Consider partitioning for very large tables
- Use appropriate index types (B-tree, Hash, etc.) based on query patterns
