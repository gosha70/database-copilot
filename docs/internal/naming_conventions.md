# Database Naming Conventions

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
