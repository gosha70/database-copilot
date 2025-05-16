"""
Streamlit application for Database Copilot with performance optimizations.
"""
import os
import logging
import sys
import tempfile
import time
import yaml
import asyncio
import yaml
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from functools import lru_cache

# Use the file watcher type from environment variable or default to "poll"
#os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = os.environ.get("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none" 

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up asyncio event loop policy to avoid "no running event loop" errors
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    # For Unix-based systems, use the default policy but ensure we have a loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Set up asyncio event loop policy to avoid "no running event loop" errors
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    # For Unix-based systems, use the default policy but ensure we have a loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Import custom styles
from backend.static.styles import CUSTOM_CSS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Lazy loading of models and components
_COMPONENTS = {}

def get_component(component_name: str, debug_mode: bool = False) -> Any:
    """
    Lazily load and cache components to reduce startup time and memory usage.
    
    Args:
        component_name: Name of the component to load.
        debug_mode: Whether to enable debug mode for the component.
        
    Returns:
        The loaded component.
    """
    # Create a unique key for the component with debug mode and LLM type
    llm_type = os.environ.get("LLM_TYPE", "local")
    component_key = f"{component_name}_{debug_mode}_{llm_type}"
    
    # Check if we need to reload the component due to settings changes
    if "previous_settings" not in st.session_state:
        st.session_state.previous_settings = {
            "debug_mode": debug_mode,
            "llm_type": llm_type
        }
    
    # If settings have changed, clear the cached component
    if (st.session_state.previous_settings["debug_mode"] != debug_mode or 
        st.session_state.previous_settings["llm_type"] != llm_type):
        if component_key in _COMPONENTS:
            logger.info(f"Settings changed, reloading component: {component_name}")
            del _COMPONENTS[component_key]
        # Update previous settings
        st.session_state.previous_settings = {
            "debug_mode": debug_mode,
            "llm_type": llm_type
        }
    
    if component_key in _COMPONENTS:
        return _COMPONENTS[component_key]
    
    start_time = time.time()
    logger.info(f"Loading component: {component_name} (debug_mode={debug_mode}, llm_type={llm_type})")
    
    if component_name == "parser":
        from backend.models.liquibase_parser import LiquibaseParser
        _COMPONENTS[component_key] = LiquibaseParser()
    elif component_name == "reviewer":
        from backend.models.liquibase_reviewer import LiquibaseReviewer
        _COMPONENTS[component_key] = LiquibaseReviewer(debug_mode=debug_mode)
    elif component_name == "generator":
        from backend.models.liquibase_generator import LiquibaseGenerator
        _COMPONENTS[component_key] = LiquibaseGenerator(debug_mode=debug_mode)
    elif component_name == "qa_system":
        from backend.models.qa_system import QASystem
        _COMPONENTS[component_key] = QASystem()
    elif component_name == "entity_generator":
        from backend.models.entity_generator import EntityGenerator
        _COMPONENTS[component_key] = EntityGenerator()
    elif component_name == "test_generator":
        from backend.models.test_generator import TestGenerator
        _COMPONENTS[component_key] = TestGenerator()
    else:
        raise ValueError(f"Unknown component: {component_name}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Loaded component {component_name} in {elapsed_time:.2f} seconds")
    
    return _COMPONENTS[component_key]

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

def review_migration_file(file_path: str, debug_mode: bool = False) -> str:
    """
    Review a Liquibase migration file.
    
    Args:
        file_path: The path to the migration file.
        debug_mode: Whether to enable debug mode.
    
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
        
        # Lazy load the reviewer with debug mode
        reviewer = get_component("reviewer", debug_mode=debug_mode)
        
        # Review the migration
        review = reviewer.review_migration(migration_content, format_type)
        return review
    except Exception as e:
        logger.error(f"Error reviewing migration file: {e}")
        return f"Error reviewing migration file: {str(e)}"

def generate_migration(description: str, format_type: str, author: str, debug_mode: bool = False) -> str:
    """
    Generate a Liquibase migration from a natural language description.
    
    Args:
        description: Natural language description of the migration.
        format_type: The format of the migration file (xml or yaml).
        author: The author of the migration.
        debug_mode: Whether to enable debug mode.
    
    Returns:
        A Liquibase migration.
    """
    try:
        # Lazy load the generator with debug mode
        generator = get_component("generator", debug_mode=debug_mode)
        
        # Generate the migration
        migration = generator.generate_migration(description, format_type, author)
        return migration
    except Exception as e:
        logger.error(f"Error generating migration: {e}")
        return f"Error generating migration: {str(e)}"

@lru_cache(maxsize=1)
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
    Main function to run the Streamlit application.
    """
    start_time = time.time()
    logger.info("Starting Database Copilot application")
    
    # Ensure secrets file exists
    ensure_secrets_file()
    
    # Check if vector store initialization should be disabled
    if os.environ.get("DISABLE_VECTOR_STORE_INIT", "0") == "1":
        logger.info("Vector store initialization disabled")
        # Set a flag to indicate that vector store is disabled
        st.session_state.vector_store_disabled = True
    
    st.set_page_config(
        page_title="Database Copilot",
        page_icon="🗃️", 
        layout="wide"
    )
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
            
    # Create sidebar with appearance settings
    with st.sidebar:
        st.markdown("## Appearance Settings")
        
        # Add debug mode toggle
        debug_mode = st.checkbox("Enable Debug Mode", value=False, key="debug_mode")
        if debug_mode:
            st.info("Debug mode enabled. Check the terminal for detailed logs.")
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        
        # Add dark mode toggle
        st.markdown("### Theme")
        theme_mode = st.selectbox(
            "Select theme",
            ["Light", "Dark"],
            index=0
        )
        
        # Add color pickers
        st.markdown("### Colors")
        primary_color = st.color_picker("Primary Color", "#295ED2")
        secondary_color = st.color_picker("Secondary Color", "#2196F3")
        text_color = st.color_picker("Text Color", "#77E5FF")


        # Apply theme based on selection
        if theme_mode == "Dark":
            st.markdown("""
            <style id="dark_theme">
                /* Root variables */
            <style id="dark_theme">
                /* Root variables */
                :root {
                    --background-color: #121212;
                    --text-color: #E0E0E0;
                    --secondary-background-color: #1E1E1E;
                    --border-color: #333333;
                    --widget-background: #3A3A3A;
                    --widget-border: #5684EA;
                    --checkbox-background: #2196F3;
                }
                
                /* Main app background */
                .stApp {
                    background-color: var(--background-color) !important;
                    color: var(--text-color) !important;
                }
                
                /* Sidebar */
                .stSidebar {
                    background-color: var(--secondary-background-color) !important;
                    border-right: 1px solid var(--border-color) !important;
                }
                
                /* Text inputs */
                .stTextInput > div > div > input {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Text areas */
                .stTextArea > div > div > textarea {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Select boxes */
                .stSelectbox > div > div {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Dropdowns */
                .stSelectbox > div > div > div {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* Expanders */
                .streamlit-expanderHeader {
                    background-color: var(--secondary-background-color) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                .streamlit-expanderContent {
                    background-color: var(--background-color) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Code blocks */
                .stCodeBlock {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* Ensure code text is visible */
                .stCodeBlock code {
                    color: white !important;
                }
                
                /* Tabs */
                .stTabs [data-baseweb="tab-list"] {
                    background-color: var(--secondary-background-color) !important;
                    border-bottom: 1px solid var(--widget-border) !important;
                }
                
                .stTabs [data-baseweb="tab"] {
                    color: var(--text-color) !important;
                }
                
                /* Buttons */
                .stButton > button {
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Checkboxes */
                .stCheckbox > div > div > label {
                    color: var(--text-color) !important;
                }
                
                /* Checkbox background color */
                .stCheckbox > div > div > div[data-baseweb="checkbox"] > div {
                    background-color: var(--checkbox-background) !important;
                }
                
                /* File uploader */
                .stFileUploader > div {
                    background-color: var(--widget-background) !important;
                    color: white !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* File uploader drop area */
                .stFileUploader > div > div:first-child {
                    background-color: var(--widget-background) !important;
                    color: white !important;
                    border: 1px dashed var(--widget-border) !important;
                }
                
                /* File uploader text */
                .stFileUploader p, 
                .stFileUploader span, 
                .stFileUploader div {
                    color: white !important;
                }
                
                /* Dataframes */
                .stDataFrame {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* Table */
                .stTable {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* All text */
                p, h1, h2, h3, h4, h5, h6, li, span, div {
                    color: var(--text-color) !important;
                }
                
                /* All labels */
                label {
                    color: var(--text-color) !important;
                }
            </style>
            """, unsafe_allow_html=True)
        else:
            # Light theme - explicitly set light mode styles and remove any dark mode styles
            st.markdown("""
            <style id="light_theme">
                /* Root variables */
                :root {
                    --background-color: #FFFFFF;
                    --text-color: #24cbd1;
                    --secondary-background-color: #F0F2F6;
                    --border-color: #CCCCCC;
                    --widget-background: #072B7D;
                    --widget-border: #DDDDDD;
                }
                
                /* Main app background */
                .stApp {
                    background-color: var(--background-color) !important;
                    color: var(--text-color) !important;
                    background-color: var(--background-color) !important;
                    color: var(--text-color) !important;
                }
                
                /* Sidebar */
                /* Sidebar */
                .stSidebar {
                    background-color: var(--secondary-background-color) !important;
                    border-right: 1px solid var(--border-color) !important;
                    background-color: var(--secondary-background-color) !important;
                    border-right: 1px solid var(--border-color) !important;
                }
                
                /* Text inputs */
                /* Text inputs */
                .stTextInput > div > div > input {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Text areas */
                /* Text areas */
                .stTextArea > div > div > textarea {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Select boxes */
                .stSelectbox > div > div {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Dropdowns */
                .stSelectbox > div > div > div {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* Expanders */
                .streamlit-expanderHeader {
                    background-color: var(--secondary-background-color) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                .streamlit-expanderContent {
                    background-color: var(--background-color) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Code blocks */
                .stCodeBlock {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* Tabs */
                .stTabs [data-baseweb="tab-list"] {
                    background-color: var(--secondary-background-color) !important;
                    border-bottom: 1px solid var(--widget-border) !important;
                }
                
                .stTabs [data-baseweb="tab"] {
                    color: var(--text-color) !important;
                }
                
                /* Buttons */
                .stButton > button {
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Checkboxes */
                .stCheckbox > div > div > label {
                    color: var(--text-color) !important;
                }
                
                /* File uploader */
                .stFileUploader > div {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Dataframes */
                .stDataFrame {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* Table */
                .stTable {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* All text */
                p, h1, h2, h3, h4, h5, h6, li, span, div {
                    color: var(--text-color) !important;
                }
                
                /* All labels */
                label {
                    color: var(--text-color) !important;
                    background-color: transparent !important;
                    border: none !important;
                }
                
                /* Select boxes */
                .stSelectbox > div > div {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Dropdowns */
                .stSelectbox > div > div > div {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* Expanders */
                .streamlit-expanderHeader {
                    background-color: var(--secondary-background-color) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                .streamlit-expanderContent {
                    background-color: var(--background-color) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Code blocks */
                .stCodeBlock {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* Tabs */
                .stTabs [data-baseweb="tab-list"] {
                    background-color: var(--secondary-background-color) !important;
                    border-bottom: 1px solid var(--widget-border) !important;
                }
                
                .stTabs [data-baseweb="tab"] {
                    color: var(--text-color) !important;
                }
                
                /* Buttons */
                .stButton > button {
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Checkboxes */
                .stCheckbox > div > div > label {
                    color: var(--text-color) !important;
                }
                
                /* File uploader */
                .stFileUploader > div {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                    border: 1px solid var(--widget-border) !important;
                }
                
                /* Dataframes */
                .stDataFrame {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* Table */
                .stTable {
                    background-color: var(--widget-background) !important;
                    color: var(--text-color) !important;
                }
                
                /* All text */
                p, h1, h2, h3, h4, h5, h6, li, span, div {
                    color: var(--text-color) !important;
                }
                
                /* All labels */
                label {
                    color: var(--text-color) !important;
                }
            </style>
            """, unsafe_allow_html=True)
        
        # Apply selected colors using custom CSS
        custom_css = f"""
        <style>
        /* Fix checkbox label highlighting */
        [data-testid="stCheckbox"] label {{
            background-color: transparent !important;
            color: var(--text-color) !important;
        }}
        
        /* Make text in text areas and inputs black for better visibility in light mode */
        .stTextArea textarea, .stTextInput input {{
            color: #000000 !important;
            font-weight: bold !important;
        }}
        
        /* Make dropdown text black for better visibility */
        .stSelectbox [data-baseweb="select"] div, 
        .stSelectbox [data-baseweb="popover"] div,
        [data-baseweb="menu"] ul li,
        [data-baseweb="popover"] ul li,
        [data-baseweb="select"] ul li,
        [role="listbox"] li {{
            color: #000000 !important;
            font-weight: bold !important;
            background-color: {secondary_color} !important;
        }}
        
        /* Make placeholder text more visible */
        ::placeholder {{
            color: #686868 !important;
            opacity: 0.7 !important;
            font-weight: normal !important;
        }}

        /* Fix checkbox label highlighting */
        [data-testid="stCheckbox"] label {{
            background-color: transparent !important;
            color: var(--text-color) !important;
        }}
        
        /* Make text in text areas and inputs black for better visibility in light mode */
        .stTextArea textarea, .stTextInput input {{
            color: #000000 !important;
            font-weight: bold !important;
        }}
        
        /* Make dropdown text black for better visibility */
        .stSelectbox [data-baseweb="select"] div, 
        .stSelectbox [data-baseweb="popover"] div,
        [data-baseweb="menu"] ul li,
        [data-baseweb="popover"] ul li,
        [data-baseweb="select"] ul li,
        [role="listbox"] li {{
            color: #000000 !important;
            font-weight: bold !important;
            background-color: {secondary_color} !important;
        }}
        
        /* Make placeholder text more visible */
        ::placeholder {{
            color: #686868 !important;
            opacity: 0.7 !important;
            font-weight: normal !important;
        }}

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
        
        /* Apply color to custom-label class */
        .custom-label {{
            color: {primary_color} !important;
            font-weight: bold !important;
            margin-bottom: 0.5rem !important;
        }}
        
        /* Target all Streamlit labels */
        label, .stSelectbox label, .stSlider label, .stCheckbox label {{
            color: {primary_color} !important;
            font-weight: bold !important;
            background-color: transparent !important;
            border: none !important;
        }}
        
        /* Target specific label types that might be harder to style */
        [data-baseweb="select"] + div, 
        [data-baseweb="base-input"] + div,
        .stColorPicker label,
        .stFileUploader label {{
            color: {primary_color} !important;
            font-weight: bold !important;
        }}
        
        /* Target sidebar section headers */
        .sidebar .stMarkdown h1, 
        .sidebar .stMarkdown h2, 
        .sidebar .stMarkdown h3 {{
            color: {primary_color} !important;
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
        
        # Add performance settings
        st.markdown("## Performance Settings")
        
        # Add option to enable/disable vector store
        if st.session_state.get("vector_store_disabled", False):
            if st.button("Enable Vector Store"):
                st.session_state.vector_store_disabled = False
                st.success("Vector store enabled. Reload the page to apply changes.")
        else:
            if st.button("Disable Vector Store"):
                st.session_state.vector_store_disabled = True
                st.success("Vector store disabled. Reload the page to apply changes.")
        
        # Add option to use external LLM
        use_external_llm = st.checkbox("Use External LLM", value=os.environ.get("LLM_TYPE", "local") != "local")
        if use_external_llm:
            os.environ["LLM_TYPE"] = "openai"  # Default to OpenAI
            llm_provider = st.selectbox(
                "LLM Provider",
                ["openai", "claude", "gemini", "mistral", "deepseek"],
                index=0
            )
            os.environ["LLM_TYPE"] = llm_provider
            st.success(f"Using external LLM: {llm_provider}. All components will use this LLM.")
            
            # Check if API key is set
            api_key_env_var = f"{llm_provider.upper()}_API_KEY"
            if llm_provider == "claude":
                api_key_env_var = "ANTHROPIC_API_KEY"
            elif llm_provider == "gemini":
                api_key_env_var = "GOOGLE_API_KEY"
                
            if not os.environ.get(api_key_env_var):
                st.warning(f"No API key found for {llm_provider}. Please set {api_key_env_var} in .streamlit/secrets.toml")
        else:
            os.environ["LLM_TYPE"] = "local"
            st.info("Using local LLM. All components will use the local model.")
    
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
                        # First check if the file can be parsed
                        try:
                            # Get the file format
                            format_type = get_file_format(file_path)
                            if format_type == "yaml":
                                # Try to parse the YAML file
                                with open(file_path, 'r') as f:
                                    yaml_content = f.read()
                                try:
                                    yaml.safe_load(yaml_content)
                                except yaml.YAMLError as e:
                                    st.error(f"Invalid YAML file. Please fix the following error and try again:\n\n```\n{str(e)}\n```")
                                    st.stop()
                            
                            # If we get here, the file is valid, so proceed with review
                            review = review_migration_file(file_path, debug_mode=debug_mode)
                            
                            with st.expander("Review Results", expanded=True):
                                st.markdown(review)
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
                
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
                    migration = generate_migration(description, format_type, author, debug_mode=debug_mode)
                
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
            ["all", "jpa", "liquibase", "internal", "examples", "java"],
            index=0,
            help="Select the category of documentation to search in."
        )
        
        # Answer button
        if st.button("Answer Question", key="answer_button"):
            if not question:
                st.error("Please provide a question.")
            else:
                with st.spinner("Answering question..."):
                    # Lazy load the QA system
                    qa_system = get_component("qa_system")
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
                        # Lazy load the entity generator
                        entity_generator = get_component("entity_generator")
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
                    # Lazy load the test generator
                    test_generator = get_component("test_generator")
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
    
    # Display startup time
    elapsed_time = time.time() - start_time
    logger.info(f"Application started in {elapsed_time:.2f} seconds")
    
    # Add startup time to footer
    st.markdown(f"""
    <div style="position: fixed; bottom: 0; right: 0; padding: 10px; background-color: rgba(0,0,0,0.1); border-radius: 5px; font-size: 0.8em;">
        Startup time: {elapsed_time:.2f}s
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
