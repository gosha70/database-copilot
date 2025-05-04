"""
Script to run the FastAPI application for Database Copilot.
"""
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the FastAPI application.
    """
    try:
        from backend.api import run_api
        logger.info("Starting Database Copilot API server")
        run_api()
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
