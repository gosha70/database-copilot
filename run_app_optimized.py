"""
Script to run the Database Copilot application with optimized resource usage.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ensure_secrets_file():
    """
    Ensure that a secrets.toml file exists.
    """
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

def main():
    """
    Main function to run the application with optimized resource usage.
    """
    parser = argparse.ArgumentParser(description="Run the Database Copilot application")
    parser.add_argument("--cpu-limit", type=int, default=None, 
                        help="Limit CPU usage (number of cores)")
    parser.add_argument("--memory-limit", type=int, default=None,
                        help="Limit memory usage in MB")
    parser.add_argument("--use-external-llm", action="store_true",
                        help="Use external LLM API instead of local model")
    parser.add_argument("--disable-vector-store", action="store_true",
                        help="Disable vector store initialization at startup")
    parser.add_argument("--lazy-load", action="store_true",
                        help="Lazy load models only when needed")
    
    args = parser.parse_args()
    
    # Ensure secrets file exists
    ensure_secrets_file()
    
    # Set environment variables for optimization
    if args.use_external_llm:
        os.environ["LLM_TYPE"] = "openai"  # Change to your preferred external LLM
        logger.info("Using external LLM API instead of local model")
    
    # Set CPU limit if specified
    if args.cpu_limit:
        cpu_count = min(args.cpu_limit, multiprocessing.cpu_count())
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
        logger.info(f"Limited CPU usage to {cpu_count} cores")
    
    # Set memory limit if specified
    if args.memory_limit:
        # Convert MB to bytes
        memory_limit_bytes = args.memory_limit * 1024 * 1024
        # This is a soft limit that helps PyTorch manage memory better
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.memory_limit}"
        logger.info(f"Limited memory usage to {args.memory_limit} MB")
    
    # Disable vector store initialization at startup if specified
    if args.disable_vector_store:
        os.environ["DISABLE_VECTOR_STORE_INIT"] = "1"
        logger.info("Disabled vector store initialization at startup")
    
    # Enable lazy loading if specified
    if args.lazy_load:
        os.environ["LAZY_LOAD_MODELS"] = "1"
        logger.info("Enabled lazy loading of models")
    
    # Set Streamlit to use polling file watcher which is less resource-intensive
    #os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    
    # Run the Streamlit app with the optimized version
    logger.info("Starting Database Copilot application with optimizations")
    os.system(f"{sys.executable} -m streamlit run backend/app_optimized.py")

if __name__ == "__main__":
    main()
