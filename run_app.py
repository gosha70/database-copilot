"""
Script to run the Database Copilot application.
"""
import os
import sys
import logging
from pathlib import Path

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
    Main function to run the application.
    """
    # Ensure secrets file exists
    ensure_secrets_file()
    
    # Run the Streamlit app
    logger.info("Starting Database Copilot application")
    os.system(f"{sys.executable} -m streamlit run backend/app_optimized.py")

if __name__ == "__main__":
    main()
