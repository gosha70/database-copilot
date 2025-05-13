#!/usr/bin/env python3
"""
Script to download the nomic-embed-text-v1.5.Q4_K_M.gguf embedding model.

This script downloads the model and saves it to the data/hf_models directory.
"""
import os
import logging
import argparse
import sys
from pathlib import Path

# Check for required dependencies
missing_deps = []
try:
    import requests
except ImportError:
    missing_deps.append("requests")

try:
    from tqdm import tqdm
except ImportError:
    missing_deps.append("tqdm")

# If dependencies are missing, print instructions and exit
if missing_deps:
    print("Error: Missing required dependencies:")
    for dep in missing_deps:
        print(f"  - {dep}")
    print("\nPlease install the missing dependencies with one of these commands:")
    print(f"  pip install {' '.join(missing_deps)}")
    print(f"  pip3 install {' '.join(missing_deps)}")
    print(f"  python -m pip install {' '.join(missing_deps)}")
    print(f"  python3 -m pip install {' '.join(missing_deps)}")
    print("\nIf you're using a virtual environment, make sure it's activated first.")
    print("If you're using conda, make sure the conda environment is activated first.")
    print("\nAlternatively, you can run the setup script which will install all dependencies:")
    print("  ./run_torch_free.sh setup")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_file(url: str, output_path: str) -> None:
    """
    Download a file with progress bar.
    
    Args:
        url: The URL to download from.
        output_path: The path to save the file to.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as file, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    """
    Main function to download the embedding model.
    """
    parser = argparse.ArgumentParser(description='Download nomic-embed-text model')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the model to (default: data/hf_models)')
    
    args = parser.parse_args()
    
    # Set up paths
    model_url = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"
    
    # Set the output directory
    if args.output_dir is None:
        # Use the default directory in the project
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = script_dir  # Assuming the script is in the project root
        output_dir = os.path.join(project_root, "data", "hf_models")
    else:
        output_dir = args.output_dir
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the output path
    output_path = os.path.join(output_dir, "nomic-embed-text-v1.5.Q4_K_M.gguf")
    
    # Check if the model already exists
    if os.path.exists(output_path):
        logger.info(f"Model already exists at {output_path}")
        return
    
    # Download the model
    logger.info(f"Downloading nomic-embed-text-v1.5.Q4_K_M.gguf to {output_path}")
    logger.info(f"This file is approximately 180 MB and may take a few minutes to download")
    
    try:
        download_file(model_url, output_path)
        logger.info(f"Download complete: {output_path}")
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        # Remove partial download if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        return 1
    
    # Verify the download
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        logger.info(f"Downloaded file size: {file_size / (1024 * 1024):.2f} MB")
    else:
        logger.error("Download failed: File not found")
        return 1
    
    logger.info("You can now use this model with the Database Copilot")
    logger.info("Set EMBEDDING_TYPE=llama_cpp in your environment or .streamlit/secrets.toml")
    
    return 0

if __name__ == "__main__":
    exit(main())
