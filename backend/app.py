"""
Streamlit application for Database Copilot.
"""
import os
import logging
import sys
import tempfile
from typing import Optional
from pathlib import Path
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"   # safest

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Import custom styles
from backend.static.styles import CUSTOM_CSS

from backend.models.liquibase_parser import LiquibaseParser
from backend.models.liquibase_reviewer import LiquibaseReviewer
from backend.models.liquibase_generator import LiquibaseGenerator
from backend.models.qa_system import QASystem
from backend.models.entity_generator import EntityGenerator
from backend.models.test_generator import TestGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize parser, reviewer, generator, QA system, entity generator, and test generator
parser = LiquibaseParser()
reviewer = LiquibaseReviewer()
generator = LiquibaseGenerator()
qa_system = QASystem()
entity_generator = EntityGenerator()
test_generator = TestGenerator()

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

def get_image_path():
    """
    Get the path to the logo image.
    
    Returns:
        Path to the logo image, or None if no image is found.
    """
    # Check for image in the static/images directory
    image_dir = Path(__file__).parent / "static" / "images"
    
    # Look for any image file with "logo" in the name
    logo_files = list(image_dir.glob("*logo*.*"))
    
    # If no logo files found, look for any image file
    if not logo_files:
        logo_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
    
    # Return the first image found, or None if no images found
    return logo_files[0] if logo_files else None

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(
        page_title="Database Copilot",
        page_icon="🗄️",
        layout="wide"
    )
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Create sidebar with appearance settings
    with st.sidebar:
        st.markdown("## Appearance Settings")
        
        # Add dark mode toggle
        st.markdown("### Theme")
        theme_mode = st.selectbox(
            "Select theme",
            ["Light", "Dark"],
            index=0
        )
        
        # Add color pickers
        st.markdown("### Colors")
        primary_color = st.color_picker("Primary Color", "#4CAF50")
        secondary_color = st.color_picker("Secondary Color", "#2196F3")
        text_color = st.color_picker("Text Color", "#333333")
        
        # Apply theme based on selection
        if theme_mode == "Dark":
            st.markdown("""
            <style>
                :root {
                    --background-color: #121212;
                    --text-color: #E0E0E0;
                    --secondary-background-color: #1E1E1E;
                }
                
                .stApp {
                    background-color: var(--background-color);
                    color: var(--text-color);
                }
                
                .stSidebar {
                    background-color: var(--secondary-background-color);
                }
                
                .stTextInput > div > div > input {
                    background-color: var(--secondary-background-color);
                    color: var(--text-color);
                }
                
                .stTextArea > div > div > textarea {
                    background-color: var(--secondary-background-color);
                    color: var(--text-color);
                }
            </style>
            """, unsafe_allow_html=True)
        
        # Apply selected colors using custom CSS
        custom_css = f"""
        <style>
        /* Apply primary color to various elements */
        .stButton > button {{
            background-color: {primary_color} !important;
            color: white !important;
            border-color: {primary_color} !important;
        }}
        
        .streamlit-expanderHeader {{
            color: {primary_color} !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {primary_color} !important;
            color: white !important;
        }}
        
        /* Fix the tab indicator line color */
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: {primary_color} !important;
        }}
        
        /* Apply secondary color */
        .stFileUploader > div > button {{
            background-color: {secondary_color} !important;
        }}
        
        /* Apply text color */
        body {{
            color: {text_color} !important;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: {text_color} !important;
        }}
        
        /* Fix tab text color when selected and not selected */
        .stTabs [aria-selected="true"] p {{
            color: white !important;
        }}
        
        /* Fix non-selected tab text color */
        .stTabs [data-baseweb="tab"] p {{
            color: {text_color} !important;
        }}
        
        /* Fix the tab text color for the "Generate Entity" tab specifically */
        .stTabs [data-baseweb="tab"]:nth-child(4) p {{
            color: {text_color} !important;
        }}
        
        /* Override any default Streamlit tab styling */
        .stTabs [data-baseweb="tab"] {{
            color: {text_color} !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
        }}
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
    
    # Display logo and title
    col1, col2 = st.columns([1, 6])
    with col1:
        # Check if logo image exists
        logo_path = get_image_path()
        if logo_path:
            st.image(str(logo_path), width=80)
        else:
            # Default to emoji if no image is found
            st.markdown("<h1 style='font-size: 3rem; text-align: center;'>🗄️</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='margin-top: 0; margin-left: -20px;'>Database Copilot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='margin-left: -20px;'>A RAG-based assistant for database migrations and ORM in Java</p>", unsafe_allow_html=True)
    
    # Add logo uploader to sidebar
    with st.sidebar:
        st.markdown("## Logo Settings")
        logo_file = st.file_uploader("Upload Logo", type=["png", "jpg", "jpeg"], key="logo_uploader")
        if logo_file:
            # Save the uploaded logo
            logo_dir = Path(__file__).parent / "static" / "images"
            logo_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with a consistent name
            logo_path = logo_dir / f"logo{Path(logo_file.name).suffix}"
            with open(logo_path, "wb") as f:
                f.write(logo_file.getvalue())
            
            st.success(f"Logo saved as {logo_path.name}")
            st.image(logo_file, width=100, caption="Current Logo")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Review Migration", 
        "Generate Migration", 
        "Q/A System", 
        "Generate Entity", 
        "Generate Tests"
    ])
    
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
                
                with st.expander("Migration Content", expanded=True):
                    st.code(migration_content, language=get_file_format(file_path))
                
                # Review button
                if st.button("Review Migration", key="review_button"):
                    with st.spinner("Reviewing migration..."):
                        review = review_migration_file(file_path)
                    
                    with st.expander("Review Results", expanded=True):
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
                
                with st.expander("Generated Migration", expanded=True):
                    st.code(migration, language=format_type)
                
                # Download button
                extension = ".xml" if format_type == "xml" else ".yaml"
                st.download_button(
                    label="Download Migration",
                    data=migration,
                    file_name=f"migration{extension}",
                    mime="text/plain"
                )
    
    # Tab 3: Q/A System
    with tab3:
        st.header("Q/A System for JPA/Hibernate and Liquibase")
        st.write("Ask questions about JPA/Hibernate, ORM, Liquibase, and general database concepts.")
        
        # Input fields
        question = st.text_area("Question", height=100, placeholder="Ask a question about JPA/Hibernate or Liquibase, e.g., 'What is the difference between @OneToMany and @ManyToMany in JPA?'")
        
        # Category selection
        category = st.selectbox(
            "Documentation Category",
            ["all", "jpa", "liquibase", "internal", "examples"],
            index=0,
            help="Select the category of documentation to search in."
        )
        
        # Answer button
        if st.button("Answer Question", key="answer_button"):
            if not question:
                st.error("Please provide a question.")
            else:
                with st.spinner("Answering question..."):
                    answer = qa_system.answer_question(question, category)
                
                with st.expander("Answer", expanded=True):
                    st.markdown(answer)
    
    # Tab 4: Generate Entity
    with tab4:
        st.header("Generate JPA Entity from Liquibase Migration")
        st.write("Generate a JPA entity class from a Liquibase migration file.")
        
        # Input fields
        uploaded_file = st.file_uploader("Choose a file", type=["xml", "yaml", "yml"], key="entity_file")
        
        col1, col2 = st.columns(2)
        with col1:
            package_name = st.text_input("Package Name", value="com.example.entity")
        with col2:
            lombok = st.checkbox("Use Lombok", value=True)
        
        if uploaded_file is not None:
            # Display file info
            st.write(f"Uploaded file: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Save the uploaded file
            file_path = save_uploaded_file(uploaded_file)
            
            if file_path:
                # Display the file content
                with open(file_path, "r") as f:
                    migration_content = f.read()
                
                with st.expander("Migration Content", expanded=True):
                    st.code(migration_content, language=get_file_format(file_path))
                
                # Generate button
                if st.button("Generate Entity", key="generate_entity_button"):
                    with st.spinner("Generating entity..."):
                        entity = entity_generator.generate_entity(
                            migration_content=migration_content,
                            format_type=get_file_format(file_path),
                            package_name=package_name,
                            lombok=lombok
                        )
                    
                    with st.expander("Generated Entity", expanded=True):
                        st.code(entity, language="java")
                    
                    # Download button
                    class_name = "Entity"  # Default class name
                    for line in entity.split("\n"):
                        if "class" in line and "{" in line:
                            parts = line.split("class")[1].split("{")[0].strip().split()
                            if parts:
                                class_name = parts[0]
                                break
                    
                    st.download_button(
                        label="Download Entity",
                        data=entity,
                        file_name=f"{class_name}.java",
                        mime="text/plain"
                    )
                    
                    # Store the entity for test generation
                    st.session_state.entity_content = entity
                
                # Clean up the temporary file
                try:
                    os.unlink(file_path)
                except:
                    pass
    
    # Tab 5: Generate Tests
    with tab5:
        st.header("Generate Tests for JPA Entity")
        st.write("Generate test classes for a JPA entity.")
        
        # Check if there's an entity from the previous tab
        entity_content = st.text_area(
            "Entity Content",
            value=st.session_state.get("entity_content", ""),
            height=300,
            placeholder="Paste your JPA entity class here or generate one in the 'Generate Entity' tab."
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_package_name = st.text_input("Test Package Name", value="com.example.entity.test")
        with col2:
            test_framework = st.selectbox(
                "Test Framework",
                ["junit5", "junit4", "testng"],
                index=0
            )
        with col3:
            include_repository_tests = st.checkbox("Include Repository Tests", value=True)
        
        # Generate button
        if st.button("Generate Tests", key="generate_tests_button"):
            if not entity_content:
                st.error("Please provide an entity class.")
            else:
                with st.spinner("Generating tests..."):
                    tests = test_generator.generate_test(
                        entity_content=entity_content,
                        package_name=test_package_name,
                        test_framework=test_framework,
                        include_repository_tests=include_repository_tests
                    )
                
                with st.expander("Generated Tests", expanded=True):
                    st.code(tests, language="java")
                
                # Download button
                class_name = "EntityTest"  # Default class name
                for line in entity_content.split("\n"):
                    if "class" in line and "{" in line:
                        parts = line.split("class")[1].split("{")[0].strip().split()
                        if parts:
                            class_name = parts[0] + "Test"
                            break
                
                st.download_button(
                    label="Download Tests",
                    data=tests,
                    file_name=f"{class_name}.java",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
