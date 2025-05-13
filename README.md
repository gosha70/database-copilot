# Database Copilot

A RAG-based assistant for database migrations and ORM in Java.

## Overview

Database Copilot is an AI-powered tool designed to assist developers with database migrations using Liquibase and JPA/Hibernate. It leverages Retrieval-Augmented Generation (RAG) to provide context-aware assistance for:

- Reviewing Liquibase migrations against best practices
- Generating Liquibase migrations from natural language descriptions
- Answering questions about JPA/Hibernate and Liquibase
- Generating JPA entities from Liquibase migrations
- Generating tests for JPA entities

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/database-copilot.git
   cd database-copilot
   ```

2. Run the setup script to install dependencies and download necessary documentation:
   ```bash
   python setup.py
   ```

3. The setup script will create an empty `.streamlit/secrets.toml` file if it doesn't exist. You can customize this file with your API keys if you want to use external LLMs.

4. Add any custom documents to the appropriate directories in `docs/`:
   - Liquibase documentation → `docs/liquibase/`
   - JPA/Hibernate documentation → `docs/jpa/`
   - Internal guidelines → `docs/internal/`
   - Example code and migrations → `docs/examples/`

5. Build the vector store:
   ```bash
   python build_vector_store.py
   ```

### Mac M1/M2 Users

If you encounter "cannot import name 'Tensor' from 'torch'" errors on Mac M1/M2, use our PyTorch-free setup:

```bash
# Make the script executable (if needed)
chmod +x run_torch_free.sh

# Run the setup script
./run_torch_free.sh setup
```

This script will:
1. Create a new conda environment (or virtual environment if conda is not available)
2. Install the necessary dependencies in the clean environment
3. Download the embedding model
4. Build the vector store without PyTorch dependencies

After setup completes, follow the instructions displayed to activate the environment and run the application.

## Usage

### Running the Application

Start the Streamlit web application:

```bash
python run_app.py
```

For Mac M1/M2 users with the PyTorch-free setup:

```bash
# First, activate the environment as instructed by the setup script
conda activate database_copilot_cpp  # or source database_copilot_cpp_env/bin/activate

# Then run the application
./run_torch_free.sh run
```

The application will be available at http://localhost:8501

### Running the API

Start the FastAPI server:

```bash
python run_api.py
```

The API will be available at http://localhost:8000

### Command-Line Options

#### Setup Script

```bash
python setup.py --skip-download  # Skip downloading documentation
```

#### Vector Store Building Script

```bash
python build_vector_store.py --category jpa  # Build only the JPA category
python build_vector_store.py --recreate      # Rebuild all categories
python build_vector_store.py --verbose       # Enable verbose output
```

## Features

### Review Liquibase Migrations

Upload a Liquibase migration file (XML or YAML) to review it against best practices and company guidelines.

### Generate Liquibase Migrations

Generate a Liquibase migration from a natural language description.

### Q/A System

Ask questions about JPA/Hibernate, ORM, Liquibase, and general database concepts.

### Generate JPA Entities

Generate a JPA entity class from a Liquibase migration file.

### Generate Tests

Generate test classes for a JPA entity.

## Customization

### Adding Custom Documents

You can add your own documents to the `docs/` directory to customize the knowledge base. See [docs/README.md](docs/README.md) for more information.

### External LLM Configuration

Database Copilot supports various external LLM providers. To configure an external LLM, edit the `.streamlit/secrets.toml` file.

## Advanced Configuration

For detailed information about the PyTorch-free implementation for Mac M1/M2 users, see [PYTORCH_FREE_RAG.md](PYTORCH_FREE_RAG.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
