#!/usr/bin/env python3
"""
Script to download the sentence-transformers/all-mpnet-base-v2 embedding model.

This script downloads the model and saves it to the data/hf_models directory.
"""
import os
import logging
import argparse
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model(model_name: str, output_dir: Optional[str] = None) -> str:
    """
    Download a model from the Hugging Face Hub.
    
    Args:
        model_name: The name of the model to download.
        output_dir: The directory to save the model to. If None, uses data/hf_models.
        
    Returns:
        The path to the downloaded model.
    """
    try:
        # Import required libraries
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from huggingface_hub import snapshot_download
        except ImportError as e:
            logger.error(f"Error importing required libraries: {e}")
            logger.error("Please install the required libraries with:")
            
            # Check if we're in a conda environment
            if os.environ.get('CONDA_DEFAULT_ENV'):
                conda_env = os.environ.get('CONDA_DEFAULT_ENV')
                logger.error(f"Detected conda environment: {conda_env}")
                logger.error("For conda environments, use:")
                logger.error("conda install -c pytorch pytorch")
                logger.error("pip install sentence-transformers huggingface_hub")
            else:
                logger.error("pip install torch sentence-transformers huggingface_hub")
            
            logger.error("\nAlternatively, you can use the install_dependencies.py script:")
            logger.error("python install_dependencies.py")
            return ""
        
        # Set the output directory
        if output_dir is None:
            # Use the default directory in the project
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = script_dir  # Assuming the script is in the project root
            output_dir = os.path.join(project_root, "data", "hf_models")
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the model directory
        model_dir = os.path.join(output_dir, os.path.basename(model_name))
        
        # Check if the model already exists
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            logger.info(f"Model already exists at {model_dir}")
            return model_dir
        
        logger.info(f"Downloading model {model_name} to {model_dir}")
        
        # Download the model
        # Method 1: Using SentenceTransformer
        try:
            logger.info("Downloading using SentenceTransformer...")
            model = SentenceTransformer(model_name, cache_folder=output_dir)
            logger.info(f"Model downloaded successfully using SentenceTransformer")
            return model_dir
        except Exception as e:
            logger.warning(f"Error downloading with SentenceTransformer: {e}")
            logger.warning("Trying alternative download method...")
        
        # Method 2: Using huggingface_hub
        try:
            logger.info("Downloading using huggingface_hub...")
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=output_dir,
                local_dir=model_dir
            )
            logger.info(f"Model downloaded successfully to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error downloading with huggingface_hub: {e}")
            raise
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return ""

def main():
    """
    Main function to download the embedding model.
    """
    parser = argparse.ArgumentParser(description='Download embedding model')
    parser.add_argument('--model', type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help='Model name to download (default: sentence-transformers/all-mpnet-base-v2)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the model to (default: data/hf_models)')
    
    args = parser.parse_args()
    
    # Download the model
    model_path = download_model(args.model, args.output_dir)
    
    if model_path:
        logger.info(f"Model downloaded successfully to {model_path}")
        logger.info("You can now use this model with the Database Copilot")
    else:
        logger.error("Failed to download the model")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
