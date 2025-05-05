"""
Question answering system for JPA/Hibernate and Liquibase using RAG.
"""
import logging
from typing import Dict, List, Optional, Union, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import NUM_RETRIEVAL_RESULTS
from backend.models.llm import get_llm
from backend.models.vector_store import get_retriever

logger = logging.getLogger(__name__)

class QASystem:
    """
    Question answering system for JPA/Hibernate and Liquibase using RAG.
    """
    
    def __init__(self):
        """
        Initialize the QA system.
        """
        self.llm = get_llm()
        
        # Get retrievers for different document categories
        self.liquibase_docs_retriever = get_retriever(collection_name="liquibase_docs")
        self.jpa_docs_retriever = get_retriever(collection_name="jpa_docs")
        self.internal_guidelines_retriever = get_retriever(collection_name="internal_guidelines")
        self.example_migrations_retriever = get_retriever(collection_name="example_migrations")
    
    def answer_question(self, question: str, category: str = "all") -> str:
        """
        Answer a question about JPA/Hibernate or Liquibase.
        
        Args:
            question: The question to answer.
            category: The category of documentation to search in (all, jpa, liquibase, internal, examples).
        
        Returns:
            An answer to the question.
        """
        logger.info(f"Answering question: {question} (category: {category})")
        
        # Get relevant documents based on the category
        relevant_docs = self._get_relevant_documents(question, category)
        
        # Combine all relevant documents
        context = self._combine_context(relevant_docs)
        truncated_context = context[:4000]  # Increased context size for larger model
        
        # Create the QA chain
        qa_chain = self._create_qa_chain()
        
        #logger.info(f"Question length: {len(question)}, Context length: {len(context)}")

        # Generate the answer
        answer = qa_chain.invoke({
            "question": question,
            "context": truncated_context
        })
        
        return answer
    
    def _get_relevant_documents(self, question: str, category: str) -> Dict[str, List[str]]:
        """
        Get relevant documents based on the question and category, using a priority system.
        
        Priority order:
        1. Internal Guidelines (highest priority)
        2. Example Migrations (YAML and XML)
        3. Liquibase Documentation
        4. JPA Documentation
        
        Args:
            question: The question to answer.
            category: The category of documentation to search in (all, jpa, liquibase, internal, examples).
        
        Returns:
            A dictionary of relevant documents by category.
        """
        # If a specific category is requested, only use that category
        if category != "all":
            relevant_docs = {}
            if category == "internal":
                internal_guidelines = self._get_relevant_internal_guidelines(question)
                if internal_guidelines:
                    relevant_docs["internal_guidelines"] = internal_guidelines
            elif category == "examples":
                example_migrations = self._get_relevant_example_migrations(question)
                if example_migrations:
                    relevant_docs["example_migrations"] = example_migrations
            elif category == "liquibase":
                liquibase_docs = self._get_relevant_liquibase_docs(question)
                if liquibase_docs:
                    relevant_docs["liquibase_docs"] = liquibase_docs
            elif category == "jpa":
                jpa_docs = self._get_relevant_jpa_docs(question)
                if jpa_docs:
                    relevant_docs["jpa_docs"] = jpa_docs
            return relevant_docs
        
        # For "all" category, implement cascading retrieval with priority
        relevant_docs = {}
        min_docs_threshold = 3  # Minimum number of relevant documents to consider sufficient
        total_docs = 0
        
        # 1. First priority: Internal Guidelines
        if "internal" in [category, "all"]:
            internal_guidelines = self._get_relevant_internal_guidelines(question)
            if internal_guidelines:
                relevant_docs["internal_guidelines"] = internal_guidelines
                total_docs += len(internal_guidelines)
        
        # 2. Second priority: Example Migrations (only if we don't have enough from higher priority)
        if total_docs < min_docs_threshold and "examples" in [category, "all"]:
            example_migrations = self._get_relevant_example_migrations(question)
            if example_migrations:
                relevant_docs["example_migrations"] = example_migrations
                total_docs += len(example_migrations)
        
        # 3. Third priority: Liquibase Documentation
        if total_docs < min_docs_threshold and "liquibase" in [category, "all"]:
            liquibase_docs = self._get_relevant_liquibase_docs(question)
            if liquibase_docs:
                relevant_docs["liquibase_docs"] = liquibase_docs
                total_docs += len(liquibase_docs)
        
        # 4. Fourth priority: JPA Documentation
        if total_docs < min_docs_threshold and "jpa" in [category, "all"]:
            jpa_docs = self._get_relevant_jpa_docs(question)
            if jpa_docs:
                relevant_docs["jpa_docs"] = jpa_docs
        
        return relevant_docs
    
    def _get_relevant_jpa_docs(self, question: str) -> List[str]:
        """
        Get relevant JPA documentation.
        
        Args:
            question: The question to answer.
        
        Returns:
            A list of relevant JPA documentation.
        """
        # Get relevant documents
        docs = self.jpa_docs_retriever.get_relevant_documents(question)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _get_relevant_liquibase_docs(self, question: str) -> List[str]:
        """
        Get relevant Liquibase documentation.
        
        Args:
            question: The question to answer.
        
        Returns:
            A list of relevant Liquibase documentation.
        """
        # Get relevant documents
        docs = self.liquibase_docs_retriever.get_relevant_documents(question)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _get_relevant_internal_guidelines(self, question: str) -> List[str]:
        """
        Get relevant internal guidelines.
        
        Args:
            question: The question to answer.
        
        Returns:
            A list of relevant internal guidelines.
        """
        # Get relevant documents
        docs = self.internal_guidelines_retriever.get_relevant_documents(question)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _get_relevant_example_migrations(self, question: str) -> List[str]:
        """
        Get relevant example migrations.
        
        Args:
            question: The question to answer.
        
        Returns:
            A list of relevant example migrations.
        """
        # Get relevant documents
        docs = self.example_migrations_retriever.get_relevant_documents(question)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _combine_context(self, relevant_docs: Dict[str, List[str]]) -> str:
        """
        Combine context from different sources in priority order.
        
        Priority order:
        1. Internal Guidelines (highest priority)
        2. Example Migrations (YAML and XML)
        3. Liquibase Documentation
        4. JPA Documentation
        
        Args:
            relevant_docs: A dictionary of relevant documents by category.
        
        Returns:
            Combined context.
        """
        context_parts = []
        
        # 1. First priority: Internal Guidelines
        if "internal_guidelines" in relevant_docs and relevant_docs["internal_guidelines"]:
            context_parts.append("## Internal Guidelines (Highest Priority)\n\n" + "\n\n".join(relevant_docs["internal_guidelines"]))
        
        # 2. Second priority: Example Migrations
        if "example_migrations" in relevant_docs and relevant_docs["example_migrations"]:
            context_parts.append("## Example Migrations (High Priority)\n\n" + "\n\n".join(relevant_docs["example_migrations"]))
        
        # 3. Third priority: Liquibase Documentation
        if "liquibase_docs" in relevant_docs and relevant_docs["liquibase_docs"]:
            context_parts.append("## Liquibase Documentation (Medium Priority)\n\n" + "\n\n".join(relevant_docs["liquibase_docs"]))
        
        # 4. Fourth priority: JPA Documentation
        if "jpa_docs" in relevant_docs and relevant_docs["jpa_docs"]:
            context_parts.append("## JPA/Hibernate Documentation (Lower Priority)\n\n" + "\n\n".join(relevant_docs["jpa_docs"]))
        
        return "\n\n".join(context_parts)
    
    def _create_qa_chain(self):
        """
        Create a chain for answering questions.
        
        Returns:
            A chain for answering questions.
        """
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a database expert specializing in JPA/Hibernate and Liquibase. Your task is to answer questions about these technologies based on the provided context.
        
        # Priority Order for Information Sources
        When answering, prioritize information in this order:
        1. Internal Guidelines (highest priority)
        2. Example Migrations (YAML and XML)
        3. Liquibase Documentation
        4. JPA Documentation
        5. Your general knowledge (lowest priority)
        
        Only fall back to lower priority sources if higher priority sources don't contain relevant information.
        
        # Question
        {question}
        
        # Reference Documentation and Guidelines
        {context}
        
        Please provide a detailed and accurate answer to the question. If the context doesn't contain enough information to answer the question, say so and provide the best answer you can based on your knowledge.
        
        Format your answer in Markdown with clear sections, code examples, and bullet points where appropriate.
        
        IMPORTANT: Always provide a direct, informative answer to the question. DO NOT return formatting instructions or templates. For example, if asked "What is Liquibase?", provide a complete explanation of what Liquibase is, not instructions on how to format an answer.
        """)
        
        # Create the chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
