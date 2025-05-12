#!/usr/bin/env python3
"""
Dependency installer for the Database Copilot.

This script checks for required dependencies and installs them if they're missing.
"""
import os
import sys
import subprocess
import importlib.util
from typing import List, Dict, Tuple

# List of required packages
REQUIRED_PACKAGES = [
    "langchain",
    "langchain_community",
    "langchain_core",
    "langchain_chroma",
    "pydantic",
    "transformers",
    "sentence_transformers",
    "torch",
    "accelerate",
    "huggingface_hub",
    "bitsandbytes",
    "llama-cpp-python",
    "chromadb",
    "pypdf",
    "beautifulsoup4",
    "lxml",
    "pyyaml",
    "unstructured",
    "markdown",
    "fastapi",
    "uvicorn",
    "streamlit",
    "python-dotenv",
    "tqdm",
    "requests",
    "numpy"
]

def check_package(package_name: str) -> bool:
    """
    Check if a package is installed.
    
    Args:
        package_name: The name of the package to check.
        
    Returns:
        True if the package is installed, False otherwise.
    """
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except ImportError:
        return False

def install_package(package_name: str) -> bool:
    """
    Install a package using pip.
    
    Args:
        package_name: The name of the package to install.
        
    Returns:
        True if the installation was successful, False otherwise.
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies() -> Tuple[List[str], List[str], List[str]]:
    """
    Check for required dependencies and install them if they're missing.
    
    Returns:
        A tuple of (installed, already_installed, failed) packages.
    """
    installed = []
    already_installed = []
    failed = []
    
    for package in REQUIRED_PACKAGES:
        if check_package(package.split(">=")[0]):
            already_installed.append(package)
        else:
            print(f"Installing {package}...")
            if install_package(package):
                installed.append(package)
            else:
                failed.append(package)
    
    return installed, already_installed, failed

def main():
    """
    Main function to check and install dependencies.
    """
    print("Checking for required dependencies...")
    installed, already_installed, failed = check_and_install_dependencies()
    
    print("\nDependency check complete.")
    print(f"Already installed: {len(already_installed)}")
    print(f"Newly installed: {len(installed)}")
    
    if installed:
        print("\nNewly installed packages:")
        for package in installed:
            print(f"  - {package}")
    
    if failed:
        print("\nFailed to install the following packages:")
        for package in failed:
            print(f"  - {package}")
        print("\nPlease install them manually using:")
        print(f"  pip install {' '.join(failed)}")
        return 1
    
    print("\nAll dependencies are installed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
