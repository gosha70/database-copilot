# Common Database Migration Patterns

## Adding a New Table
- Create the table with all necessary columns
- Add primary key constraint
- Add foreign key constraints
- Add unique constraints
- Add indexes
- Add check constraints
- Add default data if necessary

## Modifying an Existing Table
- Adding a column:
  - Add with NULL constraint if adding to a table with existing data
  - Add default value if appropriate
  - Add NOT NULL constraint in a separate changeset after populating data
- Removing a column:
  - Remove foreign key constraints referencing the column first
  - Remove indexes that include the column
  - Remove the column
- Renaming a column:
  - Create a new column with the new name
  - Copy data from the old column to the new column
  - Drop the old column

## Data Migration
- Use SQL changesets for data migration
- Keep data migration separate from schema changes
- Include transactions for data migrations
- Consider performance impact for large data migrations
- Add appropriate WHERE clauses to limit scope if possible

## Schema Refactoring
- Table Splitting:
  - Create new tables
  - Migrate data to new tables
  - Update application to use new tables
  - Drop old table when no longer needed
- Table Merging:
  - Create new merged table
  - Migrate data from original tables
  - Update application to use new table
  - Drop original tables when no longer needed
