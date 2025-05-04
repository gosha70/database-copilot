"""
Script to run the Database Copilot application.
"""
import os
import logging
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_streamlit_app():
    """
    Run the Streamlit application.
    """
    logger.info("Starting Database Copilot application")
    
    # Check if the application is already set up
    if not os.path.exists("data/vector_store") or not os.path.exists("docs/internal"):
        logger.warning("Application is not set up. Running setup script first.")
        setup_result = subprocess.run(
            [sys.executable, "setup.py", "--skip-download"],
            check=False
        )
        if setup_result.returncode != 0:
            logger.error("Setup failed. Please run setup.py manually.")
            sys.exit(1)
    
    # Run the Streamlit application
    logger.info("Running Streamlit application")
    try:
        subprocess.run(
            ["streamlit", "run", "backend/app.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run Streamlit application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit_app()
