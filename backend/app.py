"""
Streamlit application for Database Copilot.
"""
import os
import logging
import sys
import tempfile
from typing import Optional, Dict, Any, List, Tuple
import json

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from backend.config import STREAMLIT_PORT
from backend.models.liquibase_parser import LiquibaseParser
from backend.models.liquibase_reviewer import LiquibaseReviewer
from backend.models.liquibase_generator import LiquibaseGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize parser, reviewer, and generator
parser = LiquibaseParser()
reviewer = LiquibaseReviewer()
generator = LiquibaseGenerator()

def save_uploaded_file(uploaded_file: UploadedFile) -> Optional[str]:
    """
    Save an uploaded file to a temporary location.
    
    Args:
        uploaded_file: The uploaded file.
    
    Returns:
        The path to the saved file, or None if the file could not be saved.
    """
    try:
        # Create a temporary file with the same extension
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        return temp_file_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return None

def get_file_format(file_path: str) -> str:
    """
    Get the format of a file based on its extension.
    
    Args:
        file_path: The path to the file.
    
    Returns:
        The format of the file (xml or yaml).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".xml":
        return "xml"
    elif ext in [".yaml", ".yml"]:
        return "yaml"
    else:
        return "unknown"

def review_migration_file(file_path: str) -> str:
    """
    Review a Liquibase migration file.
    
    Args:
        file_path: The path to the migration file.
    
    Returns:
        A review of the migration.
    """
    try:
        # Get the file format
        format_type = get_file_format(file_path)
        if format_type == "unknown":
            return "Unsupported file format. Please upload an XML or YAML file."
        
        # Read the file content
        with open(file_path, "r") as f:
            migration_content = f.read()
        
        # Review the migration
        review = reviewer.review_migration(migration_content, format_type)
        return review
    except Exception as e:
        logger.error(f"Error reviewing migration file: {e}")
        return f"Error reviewing migration file: {str(e)}"

def generate_migration(description: str, format_type: str, author: str) -> str:
    """
    Generate a Liquibase migration from a natural language description.
    
    Args:
        description: Natural language description of the migration.
        format_type: The format of the migration file (xml or yaml).
        author: The author of the migration.
    
    Returns:
        A Liquibase migration.
    """
    try:
        # Generate the migration
        migration = generator.generate_migration(description, format_type, author)
        return migration
    except Exception as e:
        logger.error(f"Error generating migration: {e}")
        return f"Error generating migration: {str(e)}"

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(
        page_title="Database Copilot",
        page_icon="🛠️",
        layout="wide"
    )
    
    st.title("Database Copilot")
    st.subheader("A RAG-based assistant for database migrations and ORM in Java")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Review Migration", "Generate Migration"])
    
    # Tab 1: Review Migration
    with tab1:
        st.header("Review Liquibase Migration")
        st.write("Upload a Liquibase migration file (XML or YAML) to review it against best practices and company guidelines.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["xml", "yaml", "yml"], key="review_file")
        
        if uploaded_file is not None:
            # Display file info
            st.write(f"Uploaded file: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Save the uploaded file
            file_path = save_uploaded_file(uploaded_file)
            
            if file_path:
                # Display the file content
                with open(file_path, "r") as f:
                    migration_content = f.read()
                
                st.subheader("Migration Content")
                st.code(migration_content, language=get_file_format(file_path))
                
                # Review button
                if st.button("Review Migration", key="review_button"):
                    with st.spinner("Reviewing migration..."):
                        review = review_migration_file(file_path)
                    
                    st.subheader("Review Results")
                    st.markdown(review)
                
                # Clean up the temporary file
                try:
                    os.unlink(file_path)
                except:
                    pass
    
    # Tab 2: Generate Migration
    with tab2:
        st.header("Generate Liquibase Migration")
        st.write("Generate a Liquibase migration from a natural language description.")
        
        # Input fields
        description = st.text_area("Migration Description", height=150, placeholder="Describe the migration you want to generate, e.g., 'Create a new table called users with columns for id, username, email, and password.'")
        
        col1, col2 = st.columns(2)
        with col1:
            format_type = st.selectbox("Format", ["xml", "yaml"], index=0)
        with col2:
            author = st.text_input("Author", value="database-copilot")
        
        # Generate button
        if st.button("Generate Migration", key="generate_button"):
            if not description:
                st.error("Please provide a migration description.")
            else:
                with st.spinner("Generating migration..."):
                    migration = generate_migration(description, format_type, author)
                
                st.subheader("Generated Migration")
                st.code(migration, language=format_type)
                
                # Download button
                extension = ".xml" if format_type == "xml" else ".yaml"
                st.download_button(
                    label="Download Migration",
                    data=migration,
                    file_name=f"migration{extension}",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
