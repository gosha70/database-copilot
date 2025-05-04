"""
Script to create example internal guidelines for database migrations.
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

# Example internal guideline for database naming conventions
GUIDELINE_NAMING_CONVENTIONS = """# Database Naming Conventions

## Table Naming
- Use snake_case for table names (e.g., `user_profiles` instead of `UserProfiles` or `userprofiles`)
- Use plural nouns for table names (e.g., `users` instead of `user`)
- Prefix tables with module name if applicable (e.g., `auth_users`, `inventory_products`)
- Keep names concise but descriptive

## Column Naming
- Use snake_case for column names (e.g., `first_name` instead of `FirstName` or `firstname`)
- Use singular nouns for column names
- Use `id` as the primary key column name
- Use `<table_name_singular>_id` for foreign key columns (e.g., `user_id` in the `orders` table)
- Use `created_at` and `updated_at` for timestamp columns
- Avoid using reserved SQL keywords as column names

## Index Naming
- Use the format `idx_<table_name>_<column_name(s)>` (e.g., `idx_users_email`)
- For multi-column indexes, include all column names in order of importance (e.g., `idx_users_last_name_first_name`)

## Constraint Naming
- Primary Key: `pk_<table_name>` (e.g., `pk_users`)
- Foreign Key: `fk_<table_name>_<referenced_table_name>` (e.g., `fk_orders_users`)
- Unique Constraint: `uq_<table_name>_<column_name(s)>` (e.g., `uq_users_email`)
- Check Constraint: `ck_<table_name>_<column_name>` (e.g., `ck_products_price`)
"""

# Example internal guideline for database design principles
GUIDELINE_DESIGN_PRINCIPLES = """# Database Design Principles

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
"""

# Example internal guideline for Liquibase best practices
GUIDELINE_LIQUIBASE_BEST_PRACTICES = """# Liquibase Best Practices

## General Guidelines
- Use a consistent format for all migrations (XML or YAML)
- Include a descriptive comment for each changeset
- Keep changesets small and focused on a single logical change
- Use meaningful changeset IDs that include a timestamp or sequential number
- Always include an author for each changeset
- Use database-agnostic types and functions when possible

## Changeset Organization
- Organize changesets in logical order
- Group related changes in the same changeset
- Separate DDL and DML operations into different changesets
- Create separate migration files for different modules or features
- Name migration files with a timestamp prefix for easy ordering

## Rollback Support
- Include rollback instructions for all changesets
- Test rollback functionality before deploying to production
- Use `<rollback>` tags in XML or `rollback:` in YAML

## Version Control
- Store all migration files in version control
- Never modify a changeset that has been applied to any environment
- Create new changesets for modifications to existing database objects
- Include the Liquibase changelog in your CI/CD pipeline

## Testing
- Test migrations in a development environment before applying to higher environments
- Use Liquibase's update SQL command to review changes before applying them
- Validate changesets with Liquibase's validate command

## Security
- Do not include sensitive data in changesets
- Use property substitution for environment-specific values
- Follow the principle of least privilege for database users executing migrations
"""

# Example internal guideline for database migration workflow
GUIDELINE_MIGRATION_WORKFLOW = """# Database Migration Workflow

## Development Phase
1. Create a new feature branch from the main branch
2. Create a new Liquibase migration file for your database changes
3. Test the migration in your local development environment
4. Commit the migration file to your feature branch
5. Create a pull request for code review

## Review Phase
1. Another developer reviews the migration file
2. The reviewer checks for adherence to naming conventions and best practices
3. The reviewer verifies that the migration includes rollback instructions
4. The reviewer tests the migration in their local environment
5. Once approved, the pull request can be merged

## Testing Phase
1. The CI/CD pipeline automatically applies the migration to the test environment
2. Automated tests are run to verify the migration
3. Manual testing is performed if necessary
4. If issues are found, create a new migration to fix them (do not modify the original)

## Deployment Phase
1. Schedule the migration for the next deployment window
2. Generate and review the update SQL before applying
3. Apply the migration to the staging environment and verify
4. Apply the migration to the production environment during the deployment window
5. Verify the migration was successful in production

## Emergency Fixes
1. For urgent production issues, follow an expedited process
2. Create a hotfix branch from the production branch
3. Create a new migration file for the fix
4. Follow an expedited review process
5. Apply the fix to production after approval
6. Merge the hotfix back to the main branch
"""

# Example internal guideline for common database migration patterns
GUIDELINE_MIGRATION_PATTERNS = """# Common Database Migration Patterns

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
"""

def save_guideline_to_file(content: str, file_path: str) -> None:
    """
    Save guideline content to a file.
    
    Args:
        content: Content to save.
        file_path: Path to save the file to.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved guideline to {file_path}")
    except Exception as e:
        logger.error(f"Error saving guideline to {file_path}: {e}")

def create_internal_guidelines(output_dir: str) -> None:
    """
    Create example internal guidelines for database migrations.
    
    Args:
        output_dir: Directory to save the guidelines to.
    """
    logger.info(f"Creating internal guidelines in {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save guidelines
    save_guideline_to_file(
        GUIDELINE_NAMING_CONVENTIONS,
        os.path.join(output_dir, "naming_conventions.md")
    )
    save_guideline_to_file(
        GUIDELINE_DESIGN_PRINCIPLES,
        os.path.join(output_dir, "design_principles.md")
    )
    save_guideline_to_file(
        GUIDELINE_LIQUIBASE_BEST_PRACTICES,
        os.path.join(output_dir, "liquibase_best_practices.md")
    )
    save_guideline_to_file(
        GUIDELINE_MIGRATION_WORKFLOW,
        os.path.join(output_dir, "migration_workflow.md")
    )
    save_guideline_to_file(
        GUIDELINE_MIGRATION_PATTERNS,
        os.path.join(output_dir, "migration_patterns.md")
    )
    
    logger.info(f"Finished creating internal guidelines in {output_dir}")

def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(description="Create internal guidelines for database migrations")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DOC_CATEGORIES["internal_guidelines"],
        help="Directory to save the guidelines to"
    )
    
    args = parser.parse_args()
    
    create_internal_guidelines(args.output_dir)

if __name__ == "__main__":
    main()
