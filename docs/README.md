# Document Directories for Database Copilot

This directory contains the documents used by Database Copilot's RAG (Retrieval-Augmented Generation) system. These documents are processed and stored in a vector database to provide context for the AI when answering questions or generating content.

## Directory Structure

- **liquibase/**: Documentation related to Liquibase database migration tool
- **jpa/**: Documentation related to JPA (Java Persistence API) and Hibernate
- **internal/**: Internal guidelines and best practices for database migrations
- **examples/**: Example migrations and code snippets

## Adding Custom Documents

You can add your own documents to any of these directories before building the vector store. This allows you to customize the knowledge base with your organization's specific guidelines, examples, or additional documentation.

### Supported File Types

The system supports the following file types:
- Markdown (.md)
- Text (.txt)
- PDF (.pdf)
- HTML (.html)
- Microsoft Word (.docx)

### How to Add Documents

1. Place your documents in the appropriate directory based on their content:
   - Liquibase documentation → `docs/liquibase/`
   - JPA/Hibernate documentation → `docs/jpa/`
   - Internal guidelines → `docs/internal/`
   - Example code and migrations → `docs/examples/`

2. Run the vector store building script to process the documents:
   ```bash
   python build_vector_store.py
   ```

3. Optionally, you can build only a specific category:
   ```bash
   python build_vector_store.py --category internal
   ```

### Best Practices for Custom Documents

- Use clear, descriptive filenames
- Organize content with proper headings and structure
- For best results, keep individual documents focused on a single topic
- Include examples where appropriate
- Use consistent terminology throughout your documents

## Rebuilding the Vector Store

If you update or add documents after initially building the vector store, you can rebuild it using the `--recreate` flag:

```bash
python build_vector_store.py --recreate
```

This will reprocess all documents and update the vector store with the latest content.
