# Database Copilot

A local standalone RAG-based application for database migrations and ORM in Java, focusing on Liquibase and JPA/Hibernate.

## Overview

Database Copilot is an AI-powered assistant that helps developers with database migrations and ORM in Java. It uses Retrieval-Augmented Generation (RAG) to provide context-aware assistance for Liquibase migrations and JPA/Hibernate entities.

The application runs completely locally, using open-source LLMs and a local vector database, ensuring that your code and data never leave your machine.

## Features

### Current Features (Increment 1)
- **Code Review of Liquibase Migrations**: Upload your XML or YAML Liquibase migrations and get detailed reviews against best practices and company guidelines.
- **Generation of Liquibase Migrations**: Generate merge-ready Liquibase migrations from natural language descriptions.

### Planned Features
- **Q/A about Databases**: Ask questions about JPA/Hibernate, ORM, Liquibase, and general database concepts (Increment 2).
- **Entity/JPA Generation**: Generate Java entity classes from Liquibase migrations (Increment 3).
- **Test Generation**: Generate test classes for your entities (Increment 3).
- **IntelliJ Plugin Integration**: Use Database Copilot directly from IntelliJ IDEA (Increment 4).

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

3. Download a test model (optional):
```bash
python download_test_model.py
```
This will download a small, quantized LLM model for testing purposes.

4. Run the application:
```bash
python run_app.py
```
This will start the Streamlit application, which you can access at http://localhost:8501.

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

## License

[MIT License](LICENSE)
