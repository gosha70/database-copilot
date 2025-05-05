# Database Copilot

A local standalone RAG-based application for database migrations and ORM in Java, focusing on Liquibase and JPA/Hibernate.

## Overview

Database Copilot is an AI-powered assistant that helps developers with database migrations and ORM in Java. It uses Retrieval-Augmented Generation (RAG) to provide context-aware assistance for Liquibase migrations and JPA/Hibernate entities.

The application runs completely locally, using open-source LLMs and a local vector database, ensuring that your code and data never leave your machine.

## Features

### Features
- **Code Review of Liquibase Migrations**: Upload your XML or YAML Liquibase migrations and get detailed reviews against best practices and company guidelines.
- **Generation of Liquibase Migrations**: Generate merge-ready Liquibase migrations from natural language descriptions.
- **Q/A about Databases**: Ask questions about JPA/Hibernate, ORM, Liquibase, and general database concepts.
- **Entity/JPA Generation**: Generate Java entity classes from Liquibase migrations.
- **Test Generation**: Generate test classes for your entities.
- **IntelliJ Plugin Integration**: Use Database Copilot directly from IntelliJ IDEA.

## Architecture

This application uses a Retrieval-Augmented Generation (RAG) approach with:
- **Python Backend**: Streamlit for the prototype UI, with plans for FastAPI in the future.
- **Local LLMs**: Uses open-source models like CodeLlama, WizardCoder, or TinyLlama for testing.
- **Local Vector Database**: ChromaDB for storing and retrieving relevant documentation and examples.

## Quick Start

### Prerequisites
- Python 3.8+
- Sufficient disk space for LLM models (varies by model, ~1GB for TinyLlama test model)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/database-copilot.git
cd database-copilot
```

2. Set up the Python environment:
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (compatible with Python 3.11)
pip install -r requirements.txt
```

3. Run the setup script:
```bash
python setup.py
```
This will:
- Create necessary directories
- Download Liquibase documentation
- Create example migrations and internal guidelines
- Ingest documents into the vector database

3. Download the LLM model:
```bash
# Download the default Mistral 7B model (recommended for better Q/A performance)
python download_test_model.py --model "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" --model-file "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Or download a smaller test model (faster but less capable)
python download_test_model.py
```
The Mistral 7B model provides significantly better performance for the Q/A system, while the default TinyLlama model is smaller and faster but may provide less accurate answers.

4. Run the application:
```bash
# Run the Streamlit UI
python run_app.py
# OR run the API server for IntelliJ plugin
python run_api.py
```
The Streamlit UI will be available at http://localhost:8501, and the API server will be available at http://localhost:8000.

## Usage

### Reviewing Liquibase Migrations

1. Navigate to the "Review Migration" tab.
2. Upload a Liquibase migration file (XML or YAML).
3. Click the "Review Migration" button.
4. The application will analyze the migration and provide a detailed review.

Example migrations are available in the `examples` directory.

### Generating Liquibase Migrations

1. Navigate to the "Generate Migration" tab.
2. Enter a natural language description of the migration you want to generate.
3. Select the format (XML or YAML) and enter an author name.
4. Click the "Generate Migration" button.
5. The application will generate a Liquibase migration based on your description.

### Q/A System

1. Navigate to the "Q/A System" tab.
2. Enter a question about JPA/Hibernate, ORM, Liquibase, or general database concepts.
3. Select the documentation category to search in.
4. Click the "Answer Question" button.
5. The application will provide a detailed answer to your question.

### Generating JPA Entities

1. Navigate to the "Generate Entity" tab.
2. Upload a Liquibase migration file (XML or YAML).
3. Enter a package name and select whether to use Lombok annotations.
4. Click the "Generate Entity" button.
5. The application will generate a JPA entity class based on the migration.

### Generating Tests

1. Navigate to the "Generate Tests" tab.
2. Enter or paste the content of a JPA entity class (or generate one in the "Generate Entity" tab).
3. Enter a package name, select a test framework, and choose whether to include repository tests.
4. Click the "Generate Tests" button.
5. The application will generate a test class for the JPA entity.

### IntelliJ Plugin

The IntelliJ plugin provides the same functionality as the web application, but integrated directly into IntelliJ IDEA.

1. Build the plugin:
```bash
cd intellij_plugin
./gradlew buildPlugin
```

2. Install the plugin in IntelliJ IDEA:
   - Open IntelliJ IDEA
   - Go to Settings/Preferences > Plugins
   - Click on the gear icon and select "Install Plugin from Disk..."
   - Navigate to the build/distributions directory and select the zip file
   - Restart IntelliJ IDEA

3. Run the API server:
```bash
python run_api.py
```

4. Use the plugin from the "Database Copilot" menu in the main menu bar or from the context menu in the Project view.

## Customizing the RAG System

You can enhance the RAG system with your own custom content to improve the quality and relevance of answers, especially for your specific use cases.

### Adding Custom Content

1. **YAML Migrations**:
   - Place your YAML migration files in `docs/examples/yaml/`
   - This directory already contains example YAML files (orders.yaml, products.yaml, users.yaml)

2. **XML Migrations**:
   - Place your XML migration files in `docs/examples/xml/`
   - This directory already contains example XML files (orders.xml, products.xml, etc.)

3. **Best Practices Documentation**:
   - Place your best practices documentation in `docs/internal/`
   - This directory already contains files like `liquibase_best_practices.md`, `migration_patterns.md`, etc.
   - Use Markdown (.md) format for best readability

4. **Java Entities**:
   - Create a new directory like `docs/entities/` for your Java entity examples
   - Alternatively, you could place them in `docs/jpa/` since they're JPA-related

5. **Java Tests**:
   - Create a new directory like `docs/tests/` for your test examples
   - Or place them alongside entities if they're closely related

### Updating the RAG System with Your Content

After placing your files in the appropriate directories, follow these steps to ensure the RAG system indexes your content:

1. **Update the configuration** (if you added new directories):
   - Edit `backend/config.py` to include any new directories you created
   - Add entries to the `DOC_CATEGORIES` dictionary, for example:
     ```python
     DOC_CATEGORIES = {
         "liquibase_docs": os.path.join(DOCS_DIR, "liquibase"),
         "internal_guidelines": os.path.join(DOCS_DIR, "internal"),
         "example_migrations": os.path.join(DOCS_DIR, "examples"),
         "jpa_docs": os.path.join(DOCS_DIR, "jpa"),
         "entity_examples": os.path.join(DOCS_DIR, "entities"),  # New directory
         "test_examples": os.path.join(DOCS_DIR, "tests"),       # New directory
     }
     ```

2. **Regenerate the vector store**:
   - Run the ingest script to process all documents and create embeddings:
     ```bash
     python -m backend.data_ingestion.ingest --recreate
     ```
   - The `--recreate` flag ensures the vector store is rebuilt from scratch
   - This will process all documents in the directories specified in `DOC_CATEGORIES`

3. **Verify ingestion**:
   - Check the logs to ensure all your documents were processed
   - Look for messages like "Successfully ingested documents into collection [collection_name]"

4. **Test the system**:
   - After ingestion, restart the application
   - Try asking questions related to your custom content to verify it's being used

### Troubleshooting

If you encounter issues with the RAG system not using your custom content:

1. **Check embedding model**: Ensure you're not seeing warnings about using `FakeEmbeddings`
2. **Verify file formats**: Make sure your files are in text-based formats that can be properly indexed
3. **Check file permissions**: Ensure the files are readable by the application
4. **Review logs**: Look for any errors during the ingestion process

### Priority System for Information Sources

The Q/A system implements a priority system that ensures internal examples and best practices take precedence over generic documentation. This helps provide more relevant and company-specific answers.

The priority order is:

1. **Internal Guidelines** (Highest Priority) - Your company's best practices and guidelines
2. **Example Migrations** (High Priority) - Your YAML and XML migration examples
3. **Liquibase Documentation** (Medium Priority) - Official Liquibase documentation
4. **JPA Documentation** (Lower Priority) - Official JPA/Hibernate documentation
5. **General Knowledge** (Lowest Priority) - The LLM's built-in knowledge

The system implements this priority in three ways:

1. **Cascading Retrieval**: The system first checks high-priority sources and only falls back to lower-priority sources if not enough relevant information is found.

2. **Context Ordering**: Information from higher-priority sources is presented first in the context provided to the LLM.

3. **Explicit Instructions**: The prompt explicitly instructs the LLM to prioritize information from higher-priority sources.

This ensures that your internal best practices and examples are given precedence when answering questions, making the system more aligned with your specific needs and standards.

## Development

### Project Structure

```
database-copilot/
├── backend/                  # Python backend code
│   ├── data_ingestion/       # Scripts to load docs into vector store
│   │   ├── document_loaders.py  # Utilities for loading different document types
│   │   ├── ingest.py         # Script to ingest documents into vector store
│   │   ├── download_liquibase_docs.py  # Script to download Liquibase docs
│   │   ├── create_example_migrations.py  # Script to create example migrations
│   │   └── create_internal_guidelines.py  # Script to create internal guidelines
│   ├── models/               # LLM integration and pipeline definitions
│   │   ├── llm.py            # LLM initialization and utilities
│   │   ├── vector_store.py   # Vector store utilities
│   │   ├── liquibase_parser.py  # Liquibase migration parser
│   │   ├── liquibase_reviewer.py  # Liquibase migration reviewer
│   │   └── liquibase_generator.py  # Liquibase migration generator
│   ├── config.py             # Configuration settings
│   └── app.py                # Streamlit application
├── docs/                     # Reference documents to index
│   ├── liquibase/            # Liquibase documentation
│   ├── internal/             # Internal guidelines
│   ├── examples/             # Example migrations
│   └── jpa/                  # JPA/Hibernate documentation (future)
├── data/                     # Persistent data and models
│   ├── vector_store/         # Local vector database
│   └── hf_models/            # Downloaded LLM and embedding models
├── examples/                 # Example files for testing
├── setup.py                  # Setup script
├── run_app.py                # Script to run the application
├── download_test_model.py    # Script to download a test model
└── README.md                 # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Roadmap

- **Increment 1**: Liquibase migration review and generation
- **Increment 2**: Q/A system for JPA/Hibernate and Liquibase
- **Increment 3**: Entity/JPA and Test generation
- **Increment 4**: IntelliJ plugin integration

### Demo
<img width="1118" alt="image" src="https://github.com/user-attachments/assets/9b2d45f9-4104-41b3-ad41-8437f4b8e511" />


## License

[MIT License](LICENSE)
