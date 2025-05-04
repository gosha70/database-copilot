"""
Test class generator for JPA entities using RAG.
"""
import logging
from typing import Dict, List, Optional, Union, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import NUM_RETRIEVAL_RESULTS
from backend.models.llm import get_llm
from backend.models.vector_store import get_retriever

logger = logging.getLogger(__name__)

class TestGenerator:
    """
    Generator for test classes for JPA entities using RAG.
    """
    
    def __init__(self):
        """
        Initialize the test generator.
        """
        self.llm = get_llm()
        
        # Get retrievers for different document categories
        self.jpa_docs_retriever = get_retriever(collection_name="jpa_docs")
        self.internal_guidelines_retriever = get_retriever(collection_name="internal_guidelines")
    
    def generate_test(
        self,
        entity_content: str,
        package_name: str = "com.example.entity.test",
        test_framework: str = "junit5",
        include_repository_tests: bool = True
    ) -> str:
        """
        Generate a test class for a JPA entity.
        
        Args:
            entity_content: The content of the entity class.
            package_name: The package name for the generated test class.
            test_framework: The test framework to use (junit5, junit4, testng).
            include_repository_tests: Whether to include repository tests.
        
        Returns:
            A test class for the JPA entity.
        """
        logger.info(f"Generating test class for JPA entity")
        
        # Extract entity information
        entity_info = self._extract_entity_info(entity_content)
        
        # Get relevant documents from different categories
        jpa_docs = self._get_relevant_jpa_docs(entity_info)
        internal_guidelines = self._get_relevant_internal_guidelines(entity_info)
        
        # Combine all relevant documents
        context = self._combine_context(jpa_docs, internal_guidelines)
        
        # Create the generation chain
        generation_chain = self._create_generation_chain()
        
        # Generate the test class
        test_class = generation_chain.invoke({
            "entity_content": entity_content,
            "entity_info": entity_info,
            "package_name": package_name,
            "test_framework": test_framework,
            "include_repository_tests": include_repository_tests,
            "context": context
        })
        
        return test_class
    
    def _extract_entity_info(self, entity_content: str) -> Dict[str, Any]:
        """
        Extract information from an entity class.
        
        Args:
            entity_content: The content of the entity class.
        
        Returns:
            A dictionary containing information about the entity.
        """
        # Extract class name
        class_name = None
        for line in entity_content.split("\n"):
            if "class" in line and "{" in line:
                parts = line.split("class")[1].split("{")[0].strip().split()
                if parts:
                    class_name = parts[0]
                    break
        
        # Extract package name
        package_name = None
        for line in entity_content.split("\n"):
            if line.strip().startswith("package"):
                package_name = line.strip().replace("package", "").replace(";", "").strip()
                break
        
        # Extract field names and types
        fields = []
        current_field = None
        for line in entity_content.split("\n"):
            line = line.strip()
            
            # Skip empty lines, comments, annotations, and non-field lines
            if not line or line.startswith("//") or line.startswith("/*") or line.startswith("*") or line.startswith("@") or "{" in line or "}" in line:
                continue
            
            # Check if line contains a field declaration
            if "private" in line or "protected" in line or "public" in line:
                parts = line.replace(";", "").split()
                if len(parts) >= 3:  # access modifier, type, name
                    field_type = parts[-2]
                    field_name = parts[-1]
                    fields.append({"name": field_name, "type": field_type})
        
        return {
            "class_name": class_name,
            "package_name": package_name,
            "fields": fields
        }
    
    def _get_relevant_jpa_docs(self, entity_info: Dict[str, Any]) -> List[str]:
        """
        Get relevant JPA documentation.
        
        Args:
            entity_info: Information about the entity.
        
        Returns:
            A list of relevant JPA documentation.
        """
        # Create a query based on the entity information
        query = f"JPA entity testing for entity: {entity_info['class_name']}"
        
        # Get relevant documents
        docs = self.jpa_docs_retriever.get_relevant_documents(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _get_relevant_internal_guidelines(self, entity_info: Dict[str, Any]) -> List[str]:
        """
        Get relevant internal guidelines.
        
        Args:
            entity_info: Information about the entity.
        
        Returns:
            A list of relevant internal guidelines.
        """
        # Create a query based on the entity information
        query = f"Internal guidelines for testing JPA entities: {entity_info['class_name']}"
        
        # Get relevant documents
        docs = self.internal_guidelines_retriever.get_relevant_documents(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _combine_context(
        self,
        jpa_docs: List[str],
        internal_guidelines: List[str]
    ) -> str:
        """
        Combine context from different sources.
        
        Args:
            jpa_docs: Relevant JPA documentation.
            internal_guidelines: Relevant internal guidelines.
        
        Returns:
            Combined context.
        """
        context_parts = []
        
        # Add JPA documentation
        if jpa_docs:
            context_parts.append("## JPA Documentation\n\n" + "\n\n".join(jpa_docs))
        
        # Add internal guidelines
        if internal_guidelines:
            context_parts.append("## Internal Guidelines\n\n" + "\n\n".join(internal_guidelines))
        
        return "\n\n".join(context_parts)
    
    def _create_generation_chain(self):
        """
        Create a chain for generating test classes.
        
        Returns:
            A chain for generating test classes.
        """
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a JPA entity test generator. Your task is to generate a test class for a JPA entity.
        
        # Entity Content
        ```java
        {entity_content}
        ```
        
        # Entity Information
        Class Name: {entity_info[class_name]}
        Package Name: {entity_info[package_name]}
        Fields: {entity_info[fields]}
        
        # Test Package Name
        {package_name}
        
        # Test Framework
        {test_framework}
        
        # Include Repository Tests
        {include_repository_tests}
        
        # Reference Documentation and Guidelines
        {context}
        
        Please generate a complete and valid test class for the JPA entity. Follow these guidelines:
        
        1. Use the specified test framework ({test_framework}).
        2. Include appropriate test annotations (@Test, @BeforeEach, etc.).
        3. Test entity creation, validation, and persistence.
        4. Include repository tests if specified.
        5. Use appropriate assertions to verify expected behavior.
        6. Follow Java naming conventions for test methods (e.g., shouldDoSomething, testSomething).
        7. Add appropriate comments to explain the purpose of each test.
        8. Include necessary imports.
        
        For JUnit 5, use:
        - @Test, @BeforeEach, @AfterEach, @BeforeAll, @AfterAll
        - Assertions.assertEquals, Assertions.assertTrue, etc.
        
        For JUnit 4, use:
        - @Test, @Before, @After, @BeforeClass, @AfterClass
        - Assert.assertEquals, Assert.assertTrue, etc.
        
        For TestNG, use:
        - @Test, @BeforeMethod, @AfterMethod, @BeforeClass, @AfterClass
        - Assert.assertEquals, Assert.assertTrue, etc.
        
        Return ONLY the Java test class without any additional explanation.
        """)
        
        # Create the chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
