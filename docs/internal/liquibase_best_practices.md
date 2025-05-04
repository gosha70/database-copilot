# Liquibase Best Practices

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
