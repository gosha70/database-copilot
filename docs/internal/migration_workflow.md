# Database Migration Workflow

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
