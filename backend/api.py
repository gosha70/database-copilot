"""
FastAPI application for Database Copilot.
"""
import os
import logging
import sys
from typing import Optional, Dict, Any, List
import tempfile

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import API_HOST, API_PORT
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

# Initialize models
parser = LiquibaseParser()
reviewer = LiquibaseReviewer()
generator = LiquibaseGenerator()
qa_system = QASystem()
entity_generator = EntityGenerator()
test_generator = TestGenerator()

# Create FastAPI app
app = FastAPI(
    title="Database Copilot API",
    description="API for Database Copilot IntelliJ Plugin",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class MigrationGenerationRequest(BaseModel):
    description: str
    format_type: str = "xml"
    author: str = "database-copilot"

class MigrationGenerationResponse(BaseModel):
    migration: str

class MigrationReviewResponse(BaseModel):
    review: str

class QuestionRequest(BaseModel):
    question: str
    category: str = "all"

class QuestionResponse(BaseModel):
    answer: str

class EntityGenerationRequest(BaseModel):
    package_name: str = "com.example.entity"
    lombok: bool = True

class EntityGenerationResponse(BaseModel):
    entity: str

class TestGenerationRequest(BaseModel):
    entity_content: str
    package_name: str = "com.example.entity.test"
    test_framework: str = "junit5"
    include_repository_tests: bool = True

class TestGenerationResponse(BaseModel):
    tests: str

@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {"message": "Welcome to Database Copilot API"}

@app.post("/api/review-migration", response_model=MigrationReviewResponse)
async def review_migration(file: UploadFile = File(...)):
    """
    Review a Liquibase migration file.
    """
    try:
        # Get the file format
        format_type = get_file_format(file.filename)
        if format_type == "unknown":
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload an XML or YAML file.")
        
        # Save the uploaded file
        temp_file_path = await save_uploaded_file(file)
        
        try:
            # Read the file content
            with open(temp_file_path, "r") as f:
                migration_content = f.read()
            
            # Review the migration
            review = reviewer.review_migration(migration_content, format_type)
            
            return {"review": review}
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Error reviewing migration: {e}")
        raise HTTPException(status_code=500, detail=f"Error reviewing migration: {str(e)}")

@app.post("/api/generate-migration", response_model=MigrationGenerationResponse)
async def generate_migration(request: MigrationGenerationRequest):
    """
    Generate a Liquibase migration from a natural language description.
    """
    try:
        # Generate the migration
        migration = generator.generate_migration(
            description=request.description,
            format_type=request.format_type,
            author=request.author
        )
        
        return {"migration": migration}
    
    except Exception as e:
        logger.error(f"Error generating migration: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating migration: {str(e)}")

@app.post("/api/answer-question", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Answer a question about JPA/Hibernate or Liquibase.
    """
    try:
        # Answer the question
        answer = qa_system.answer_question(
            question=request.question,
            category=request.category
        )
        
        return {"answer": answer}
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@app.post("/api/generate-entity", response_model=EntityGenerationResponse)
async def generate_entity(
    request: EntityGenerationRequest = Form(...),
    file: UploadFile = File(...)
):
    """
    Generate a JPA entity from a Liquibase migration.
    """
    try:
        # Get the file format
        format_type = get_file_format(file.filename)
        if format_type == "unknown":
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload an XML or YAML file.")
        
        # Save the uploaded file
        temp_file_path = await save_uploaded_file(file)
        
        try:
            # Read the file content
            with open(temp_file_path, "r") as f:
                migration_content = f.read()
            
            # Generate the entity
            entity = entity_generator.generate_entity(
                migration_content=migration_content,
                format_type=format_type,
                package_name=request.package_name,
                lombok=request.lombok
            )
            
            return {"entity": entity}
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Error generating entity: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating entity: {str(e)}")

@app.post("/api/generate-tests", response_model=TestGenerationResponse)
async def generate_tests(request: TestGenerationRequest):
    """
    Generate test classes for a JPA entity.
    """
    try:
        # Generate the tests
        tests = test_generator.generate_test(
            entity_content=request.entity_content,
            package_name=request.package_name,
            test_framework=request.test_framework,
            include_repository_tests=request.include_repository_tests
        )
        
        return {"tests": tests}
    
    except Exception as e:
        logger.error(f"Error generating tests: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating tests: {str(e)}")

async def save_uploaded_file(file: UploadFile) -> str:
    """
    Save an uploaded file to a temporary location.
    
    Args:
        file: The uploaded file.
    
    Returns:
        The path to the saved file.
    """
    # Create a temporary file with the same extension
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    return temp_file_path

def get_file_format(filename: str) -> str:
    """
    Get the format of a file based on its extension.
    
    Args:
        filename: The name of the file.
    
    Returns:
        The format of the file (xml or yaml).
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".xml":
        return "xml"
    elif ext in [".yaml", ".yml"]:
        return "yaml"
    else:
        return "unknown"

def run_api():
    """
    Run the FastAPI application.
    """
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

if __name__ == "__main__":
    run_api()
