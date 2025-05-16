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

### Dependency Compatibility

**Important:** Due to recent changes in the Python ML ecosystem, you must use the following package versions for vector store creation and OpenAI integration to work reliably:

- `sentence-transformers==2.2.2`
- `transformers==4.30.2`
- `huggingface_hub==0.15.1`
- `accelerate==0.20.3`

If you upgrade any of these packages, you may encounter ImportErrors or runtime failures. If you see errors about missing functions like `cached_download` or `split_torch_state_dict_into_shards`, or other incompatibility messages, downgrade to the versions above:

```bash
pip install sentence-transformers==2.2.2 transformers==4.30.2 huggingface_hub==0.15.1 accelerate==0.20.3
```

You can also use the provided `requirements.txt` or `backend/requirements.txt` to install compatible versions.

### Setup Options

There are two ways to set up Database Copilot:

#### Option 1: Standard Setup (Recommended for most users)

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

   If you encounter errors related to `sentence-transformers`, `transformers`, or `huggingface_hub`, see the "Dependency Compatibility" section above and ensure you have the correct versions installed.

#### Option 2: PyTorch-Free Setup (Recommended for Mac M1/M2 users or if you encounter PyTorch issues)

This setup uses llama.cpp for both embeddings and LLM inference, avoiding PyTorch dependencies:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/database-copilot.git
   cd database-copilot
   ```

2. Make the script executable (if needed):
   ```bash
   chmod +x run_torch_free.sh
   ```

3. Run the PyTorch-free setup script:
   ```bash
   ./run_torch_free.sh setup
   ```

4. After setup completes, activate the environment as instructed:
   ```bash
   # If using conda:
   conda activate database_copilot_cpp
   
   # If using venv:
   source database_copilot_cpp_env/bin/activate
   ```

5. Build the vector store:
   ```bash
   ./run_torch_free.sh rebuild
   ```

The PyTorch-free setup script will:
1. Create a new conda environment (or virtual environment if conda is not available)
2. Install the necessary dependencies in the clean environment
3. Download the embedding model
4. Provide instructions for building the vector store and running the application

## Usage

### Running the Application

#### Standard Setup

Start the Streamlit web application:

```bash
python run_app.py
```

For advanced options (CPU/memory limits, lazy loading, etc.), use:

```bash
python run_app_optimized.py [--cpu-limit N] [--memory-limit MB] [--use-external-llm] [--disable-vector-store] [--lazy-load]
```

#### PyTorch-Free Setup

Make sure you've activated the environment first:

```bash
# If using conda:
conda activate database_copilot_cpp

# If using venv:
source database_copilot_cpp_env/bin/activate
```

Then run the application:

```bash
./run_torch_free.sh run
```

The application will be available at http://localhost:8503

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

#### Troubleshooting Vector Store Build Errors

If you see errors like:

- `Could not import sentence_transformers python package.`
- `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`
- `ImportError: cannot import name 'OfflineModeIsEnabled' from 'huggingface_hub.utils'`
- `ImportError: cannot import name 'split_torch_state_dict_into_shards' from 'huggingface_hub'`

You have incompatible package versions. Run:

```bash
pip install sentence-transformers==2.2.2 transformers==4.30.2 huggingface_hub==0.15.1 accelerate==0.20.3
```

Then retry building the vector store.

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

Database Copilot supports various external LLM providers that can provide better quality reviews and recommendations. The following providers are supported:

- OpenAI (GPT-4, GPT-3.5)
- Anthropic Claude
- Google Gemini
- Mistral AI
- DeepSeek

To configure an external LLM:

1. Edit the `.streamlit/secrets.toml` file in the project root directory
2. Uncomment and set the appropriate values for your chosen LLM provider
3. Save the file and restart the application

Example for OpenAI:

```toml
# External LLM Configuration
LLM_TYPE = "openai"
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_MODEL = "gpt-4o"  # Optional, defaults to gpt-4o
```

All required packages for external LLMs are included in the project's requirements files and should be automatically installed when you run the setup script. If you encounter issues with missing packages, you can manually install them:

```bash
pip install openai anthropic google-generativeai mistralai
```

For more detailed information, refer to [external_llm_instructions.md](external_llm_instructions.md).

#### Troubleshooting External LLM Issues

If you encounter errors like "Error initializing OpenAI client" or missing package errors:

1. Make sure you've run the setup script to install all dependencies:
   ```bash
   python setup.py
   ```

2. If you're still experiencing issues, manually install the required packages:
   ```bash
   pip install openai anthropic google-generativeai mistralai
   ```

3. Verify your API key is correct in the `.streamlit/secrets.toml` file

4. Check the application logs for detailed error messages

## Advanced Configuration

For detailed information about the PyTorch-free implementation for Mac M1/M2 users, see [PYTORCH_FREE_RAG.md](PYTORCH_FREE_RAG.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
