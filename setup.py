"""
Setup script for Database Copilot.
"""
import os
import logging
import argparse
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_command(command: str) -> bool:
    """
    Run a shell command.
    
    Args:
        command: The command to run.
    
    Returns:
        True if the command succeeded, False otherwise.
    """
    logger.info(f"Running command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        return False

def create_directories() -> bool:
    """
    Create necessary directories.
    
    Returns:
        True if all directories were created successfully, False otherwise.
    """
    logger.info("Creating directories")
    try:
        # Create docs directories
        os.makedirs("docs/liquibase", exist_ok=True)
        os.makedirs("docs/internal", exist_ok=True)
        os.makedirs("docs/examples", exist_ok=True)
        os.makedirs("docs/jpa", exist_ok=True)
        
        # Create data directories
        os.makedirs("data/vector_store", exist_ok=True)
        os.makedirs("data/hf_models", exist_ok=True)
        
        # Create .streamlit directory and empty secrets.toml file if it doesn't exist
        streamlit_dir = Path(".streamlit")
        streamlit_dir.mkdir(exist_ok=True)
        
        secrets_file = streamlit_dir / "secrets.toml"
        if not secrets_file.exists():
            with open(secrets_file, "w") as f:
                f.write("# Streamlit secrets file\n")
                f.write("# Uncomment and set values as needed\n\n")
                f.write("# LLM_TYPE = \"local\"\n\n")
                f.write("# OpenAI Configuration\n")
                f.write("# OPENAI_API_KEY = \"your-openai-api-key\"\n")
                f.write("# OPENAI_MODEL = \"gpt-4o\"\n\n")
                f.write("# Claude Configuration\n")
                f.write("# ANTHROPIC_API_KEY = \"your-anthropic-api-key\"\n")
                f.write("# CLAUDE_MODEL = \"claude-3-opus-20240229\"\n\n")
                f.write("# Gemini Configuration\n")
                f.write("# GOOGLE_API_KEY = \"your-google-api-key\"\n")
                f.write("# GEMINI_MODEL = \"gemini-1.5-pro\"\n\n")
                f.write("# Mistral Configuration\n")
                f.write("# MISTRAL_API_KEY = \"your-mistral-api-key\"\n")
                f.write("# MISTRAL_MODEL = \"mistral-medium\"\n\n")
                f.write("# DeepSeek Configuration\n")
                f.write("# DEEPSEEK_API_KEY = \"your-deepseek-api-key\"\n")
                f.write("# DEEPSEEK_MODEL = \"deepseek-chat\"\n")
            logger.info(f"Created empty secrets file at {secrets_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return False

def install_dependencies() -> bool:
    """
    Install Python dependencies.
    
    Returns:
        True if dependencies were installed successfully, False otherwise.
    """
    logger.info("Installing main Python dependencies")
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt"):
        return False
    
    logger.info("Installing backend Python dependencies")
    if not run_command(f"{sys.executable} -m pip install -r backend/requirements.txt"):
        return False
    
    return True

def download_liquibase_docs() -> bool:
    """
    Download Liquibase documentation.
    
    Returns:
        True if documentation was downloaded successfully, False otherwise.
    """
    logger.info("Downloading Liquibase documentation")
    return run_command(f"{sys.executable} -m backend.data_ingestion.download_liquibase_docs")

def download_jpa_docs() -> bool:
    """
    Download JPA/Hibernate documentation.
    
    Returns:
        True if documentation was downloaded successfully, False otherwise.
    """
    logger.info("Downloading JPA/Hibernate documentation")
    return run_command(f"{sys.executable} -m backend.data_ingestion.download_jpa_docs")

def create_example_migrations() -> bool:
    """
    Create example Liquibase migrations.
    
    Returns:
        True if examples were created successfully, False otherwise.
    """
    logger.info("Creating example Liquibase migrations")
    return run_command(f"{sys.executable} -m backend.data_ingestion.create_example_migrations")

def create_internal_guidelines() -> bool:
    """
    Create internal guidelines for database migrations.
    
    Returns:
        True if guidelines were created successfully, False otherwise.
    """
    logger.info("Creating internal guidelines")
    return run_command(f"{sys.executable} -m backend.data_ingestion.create_internal_guidelines")

def ingest_documents() -> bool:
    """
    Ingest documents into the vector database.
    
    Returns:
        True if documents were ingested successfully, False otherwise.
    """
    logger.info("Ingesting documents into vector database")
    return run_command(f"{sys.executable} -m backend.data_ingestion.ingest --recreate")

def setup(skip_download: bool = False) -> bool:
    """
    Set up the Database Copilot application.
    
    Args:
        skip_download: Whether to skip downloading Liquibase documentation.
    
    Returns:
        True if setup was successful, False otherwise.
    """
    logger.info("Setting up Database Copilot")
    
    # Create directories
    if not create_directories():
        logger.error("Failed to create directories")
        return False
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Create internal guidelines
    if not create_internal_guidelines():
        logger.error("Failed to create internal guidelines")
        return False
    
    # Create example migrations
    if not create_example_migrations():
        logger.error("Failed to create example migrations")
        return False
    
    # Download documentation
    if not skip_download:
        if not download_liquibase_docs():
            logger.error("Failed to download Liquibase documentation")
            return False
        
        if not download_jpa_docs():
            logger.error("Failed to download JPA/Hibernate documentation")
            return False
    
    logger.info("Setup completed successfully")
    logger.info("To build the vector store, add any custom documents to the docs/ directories")
    logger.info("Then run: python build_vector_store.py")
    logger.info("")
    logger.info("Note: If you're using a Mac M1/M2 or encounter PyTorch-related issues,")
    logger.info("you may want to use our PyTorch-free setup instead:")
    logger.info("  ./run_torch_free.sh setup")
    return True

def main():
    """
    Main function to run the setup script.
    """
    parser = argparse.ArgumentParser(description="Set up Database Copilot")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading Liquibase documentation"
    )
    
    args = parser.parse_args()
    
    if setup(skip_download=args.skip_download):
        logger.info("Setup completed successfully")
        logger.info("To build the vector store, add any custom documents to the docs/ directories")
        logger.info("Then run: python build_vector_store.py")
        logger.info("To run the application, use: python run_app.py")
        logger.info("")
        logger.info("Alternative setup for Mac M1/M2 or if you encounter PyTorch issues:")
        logger.info("  ./run_torch_free.sh setup")
    else:
        logger.error("Setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
