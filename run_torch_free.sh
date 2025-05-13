#!/bin/bash
# Script to run Database Copilot with a PyTorch-free stack using llama.cpp

# Display banner
echo "========================================================"
echo "  Database Copilot - PyTorch-free RAG Stack"
echo "  Using llama.cpp for both LLM and embeddings"
echo "========================================================"

# Function to display usage information
show_usage() {
    echo "Usage: ./run_torch_free.sh [option]"
    echo "Options:"
    echo "  run       - Run the application with llama.cpp embeddings"
    echo "  download  - Download the embedding model only"
    echo "  rebuild   - Rebuild the vector store with llama.cpp embeddings"
    echo "  setup     - Complete setup for Mac M1/M2 (install deps, download model)"
    echo "  help      - Show this help message"
    echo ""
    echo "Example: ./run_torch_free.sh run"
}

# Check if the embedding model exists
check_model() {
    if [ ! -f "data/hf_models/nomic-embed-text-v1.5.Q4_K_M.gguf" ]; then
        echo "Embedding model not found. Downloading..."
        # Install required dependencies
        python -m pip install requests tqdm
        python download_nomic_embed.py
        if [ $? -ne 0 ]; then
            echo "Failed to download the embedding model."
            exit 1
        fi
    else
        echo "Embedding model found."
    fi
}

# Make the script executable
chmod +x download_nomic_embed.py
chmod +x rebuild_vector_store.py

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No option specified."
    show_usage
    exit 1
fi

# Process the argument
case "$1" in
    run)
        echo "Running Database Copilot with llama.cpp embeddings..."
        echo "Note: This requires all dependencies. You can install them with:"
        echo "pip install -r torch_free_requirements.txt"
        echo ""

        # Check if the embedding model exists
        check_model

        # Set environment variables
        export EMBEDDING_TYPE="llama_cpp"
        export LAZY_LOAD_MODELS="1"

        # Run the optimized app
        python run_app_optimized.py --lazy-load
        ;;
    download)
        echo "Downloading the embedding model..."
        # Install required dependencies
        python -m pip install requests tqdm
        python download_nomic_embed.py
        ;;
    rebuild)
        echo "Rebuilding the vector store with llama.cpp embeddings..."
        echo "Note: This requires additional dependencies. You can install them with:"
        echo "pip install -r torch_free_requirements.txt"
        echo ""
        
        # Install minimal dependencies
        python -m pip install requests tqdm langchain-core>=0.1.10 langchain-community>=0.0.10 langchain-chroma>=0.0.1 chromadb>=0.4.18 transformers>=4.35.0 sentence-transformers>=2.2.2 torch>=2.0.0 langchain_text_splitters>=0.0.1 beautifulsoup4>=4.12.2 lxml>=4.9.3 pypdf>=3.17.0 unstructured>=0.10.0 "dataclasses-json<0.6.0,>=0.5.7" "aiofiles<23,>=22.1.0"
        
        # Check if required packages are installed
        python rebuild_vector_store.py
        ;;
    setup)
        echo "Setting up PyTorch-free environment for Mac M1/M2..."
        
        # Inform the user about the requirements file
        echo "A torch_free_requirements.txt file has been created with all necessary dependencies."
        echo "You can install them manually with: pip install -r torch_free_requirements.txt"
        echo ""
        
        # Install minimal dependencies in the system Python environment
        # This ensures the scripts can run even if conda/venv setup fails
        echo "Installing minimal dependencies in system Python..."
        python -m pip install requests tqdm
        
        # Check if conda is available
        if command -v conda &> /dev/null; then
            echo "Creating a new conda environment for PyTorch-free setup..."
            
            # Create a new conda environment
            conda create -n database_copilot_cpp python=3.11 -y
            
            # Activate the environment
            eval "$(conda shell.bash hook)"
            conda activate database_copilot_cpp
            
            # Install dependencies in the clean environment
            echo "Installing dependencies..."
            python -m pip install requests streamlit
            python -m pip install langchain==0.1.14 langgraph==0.0.25 chromadb==0.4.24 tqdm
            python -m pip install llama-cpp-python==0.2.24
            
            # Download the embedding model
            echo "Downloading embedding model..."
            # Use the activated conda environment's Python
            python download_nomic_embed.py
            
            echo "Note: To build the vector store, run: ./run_torch_free.sh rebuild"
            
            echo "Setup complete! To run the application:"
            echo "1. Activate the environment: conda activate database_copilot_cpp"
            echo "2. Run: ./run_torch_free.sh run"
        else
            echo "Conda not found. Installing dependencies in current environment..."
            
            # Create a virtual environment using venv
            echo "Creating a virtual environment..."
            python -m venv database_copilot_cpp_env
            
            # Activate the virtual environment
            source database_copilot_cpp_env/bin/activate
            
            # Install dependencies in the clean environment
            echo "Installing dependencies..."
            python -m pip install requests streamlit
            python -m pip install langchain==0.1.14 langgraph==0.0.25 chromadb==0.4.24 tqdm
            python -m pip install llama-cpp-python==0.2.24
            
            # Download the embedding model
            echo "Downloading embedding model..."
            python download_nomic_embed.py
            
            echo "Note: To build the vector store, run: ./run_torch_free.sh rebuild"
            
            echo "Setup complete! To run the application:"
            echo "1. Activate the environment: source database_copilot_cpp_env/bin/activate"
            echo "2. Run: ./run_torch_free.sh run"
        fi
        ;;
    help)
        show_usage
        ;;
    *)
        echo "Error: Unknown option '$1'"
        show_usage
        exit 1
        ;;
esac
