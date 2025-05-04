"""
Script to download and prepare Liquibase documentation for ingestion.
"""
import os
import logging
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random

from backend.config import DOC_CATEGORIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Base URL for Liquibase documentation
LIQUIBASE_DOCS_URL = "https://docs.liquibase.com"

# List of important documentation pages to scrape
IMPORTANT_PAGES = [
    "/home.html",
    "/concepts/home.html",
    "/commands/home.html",
    "/change-types/home.html",
    "/workflows/liquibase-community/home.html",
    "/concepts/changelogs/home.html",
    "/concepts/changelogs/attributes/home.html",
    "/concepts/changelogs/xml-format.html",
    "/concepts/changelogs/yaml-format.html",
    "/change-types/create-table.html",
    "/change-types/add-column.html",
    "/change-types/drop-table.html",
    "/change-types/add-foreign-key-constraint.html",
    "/change-types/add-primary-key.html",
    "/change-types/add-unique-constraint.html",
    "/change-types/create-index.html",
    "/change-types/sql.html",
    "/best-practices/home.html",
]

def download_page(url: str) -> str:
    """
    Download a web page.
    
    Args:
        url: URL of the page to download.
    
    Returns:
        The HTML content of the page.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading page {url}: {e}")
        return ""

def extract_text_from_html(html: str) -> str:
    """
    Extract text content from HTML.
    
    Args:
        html: HTML content.
    
    Returns:
        Extracted text content.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Remove blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {e}")
        return ""

def save_text_to_file(text: str, file_path: str) -> None:
    """
    Save text content to a file.
    
    Args:
        text: Text content to save.
        file_path: Path to save the file to.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Saved text to {file_path}")
    except Exception as e:
        logger.error(f"Error saving text to {file_path}: {e}")

def download_liquibase_docs(output_dir: str) -> None:
    """
    Download Liquibase documentation.
    
    Args:
        output_dir: Directory to save the documentation to.
    """
    logger.info(f"Downloading Liquibase documentation to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download and save important pages
    for page in IMPORTANT_PAGES:
        url = urljoin(LIQUIBASE_DOCS_URL, page)
        logger.info(f"Downloading {url}")
        
        # Download page
        html = download_page(url)
        if not html:
            continue
        
        # Extract text
        text = extract_text_from_html(html)
        if not text:
            continue
        
        # Save text to file
        file_name = os.path.basename(page).replace(".html", ".txt")
        if file_name == "home.txt":
            # Use the parent directory name for home pages
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.strip("/").split("/")
            if len(path_parts) > 1:
                file_name = f"{path_parts[-2]}.txt"
        
        file_path = os.path.join(output_dir, file_name)
        save_text_to_file(text, file_path)
        
        # Sleep to avoid overloading the server
        time.sleep(random.uniform(0.5, 1.5))
    
    logger.info(f"Finished downloading Liquibase documentation to {output_dir}")

def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(description="Download Liquibase documentation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DOC_CATEGORIES["liquibase_docs"],
        help="Directory to save the documentation to"
    )
    
    args = parser.parse_args()
    
    download_liquibase_docs(args.output_dir)

if __name__ == "__main__":
    main()
