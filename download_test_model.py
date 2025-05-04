"""
Script to download a small, quantized LLM model for testing purposes.
"""
import os
import logging
import argparse
from pathlib import Path
import subprocess
import sys

from backend.config import MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default model to download
DEFAULT_MODEL = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
DEFAULT_MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Default embedding model to download
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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

def download_model(model_name: str, model_file: str, output_dir: str) -> bool:
    """
    Download a model from Hugging Face.
    
    Args:
        model_name: The name of the model on Hugging Face.
        model_file: The specific model file to download.
        output_dir: The directory to save the model to.
    
    Returns:
        True if the model was downloaded successfully, False otherwise.
    """
    logger.info(f"Downloading model {model_name}/{model_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model file already exists
    output_path = os.path.join(output_dir, model_file)
    if os.path.exists(output_path):
        logger.info(f"Model file already exists at {output_path}")
        return True
    
    # Download model file using Hugging Face CLI
    command = f"huggingface-cli download {model_name} {model_file} --local-dir {output_dir}"
    return run_command(command)

def download_embedding_model(model_name: str, output_dir: str) -> bool:
    """
    Download an embedding model from Hugging Face.
    
    Args:
        model_name: The name of the model on Hugging Face.
        output_dir: The directory to save the model to.
    
    Returns:
        True if the model was downloaded successfully, False otherwise.
    """
    logger.info(f"Downloading embedding model {model_name}")
    
    # Create output directory if it doesn't exist
    model_dir = os.path.join(output_dir, os.path.basename(model_name))
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(os.path.join(model_dir, "config.json")):
        logger.info(f"Embedding model already exists at {model_dir}")
        return True
    
    # Install huggingface_hub if not already installed
    run_command(f"{sys.executable} -m pip install huggingface_hub")
    
    # Download model using Python code
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_name, local_dir=model_dir)
        logger.info(f"Successfully downloaded embedding model to {model_dir}")
        return True
    except Exception as e:
        logger.error(f"Error downloading embedding model: {e}")
        return False

def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(description="Download a small, quantized LLM model for testing purposes")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="The name of the model on Hugging Face"
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=DEFAULT_MODEL_FILE,
        help="The specific model file to download"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="The name of the embedding model on Hugging Face"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=MODELS_DIR,
        help="The directory to save the model to"
    )
    
    args = parser.parse_args()
    
    # Install required packages
    logger.info("Installing required packages")
    run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    # Download LLM model
    if download_model(args.model, args.model_file, args.output_dir):
        logger.info(f"Successfully downloaded model to {args.output_dir}")
    else:
        logger.error("Failed to download model")
    
    # Download embedding model
    if download_embedding_model(args.embedding_model, args.output_dir):
        logger.info(f"Successfully downloaded embedding model to {args.output_dir}")
    else:
        logger.error("Failed to download embedding model")

if __name__ == "__main__":
    main()
