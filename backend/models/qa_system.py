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
from backend.models.cascade_retriever import create_cascade_retriever

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
        
        # Get retriever for Java files if the collection exists
        try:
            self.java_files_retriever = get_retriever(collection_name="java_files")
            self._java_files_available = True
        except Exception as e:
            logger.warning(f"Java files retriever not available: {e}")
            self.java_files_retriever = None
            self._java_files_available = False

        # Create CascadeRetriever for multi-source retrieval
        self.cascade_retriever = create_cascade_retriever(
            internal_guidelines_retriever=self.internal_guidelines_retriever,
            example_migrations_retriever=self.example_migrations_retriever,
            liquibase_docs_retriever=self.liquibase_docs_retriever,
            java_files_retriever=self.java_files_retriever if self._java_files_available else None,
            min_docs_per_source=3,
            max_docs_total=10
        )
    
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
        
        if category == "all":
            # Use CascadeRetriever for multi-source retrieval
            docs = self.cascade_retriever.get_relevant_documents(question)
            logger.info(f"Retrieved {len(docs)} docs: {[doc.metadata.get('source','?') for doc in docs]}")
            print("===RETRIEVED DOCS===")
            for i, doc in enumerate(docs):
                logger.info(f"Doc {i+1} (source: {doc.metadata.get('source','?')}): {doc.page_content[:200]}")
                print(f"Doc {i+1} (source: {doc.metadata.get('source','?')}): {doc.page_content[:200]}")
            print("====================")
            context = "\n\n".join([doc.page_content for doc in docs])
            truncated_context = context[:2000]
        else:
            # Use category-specific retrieval for advanced use
            relevant_docs = self._get_relevant_documents(question, category)
            context = self._combine_context(relevant_docs)
            truncated_context = context[:4000]

        logger.info(f"Context sent to LLM (first 1000 chars): {truncated_context[:1000]}")
        print("===CONTEXT SENT TO LLM===")
        print(truncated_context[:1000])
        print("=========================")
        qa_chain = self._create_qa_chain()
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
        3. Java Files (if available)
        4. Liquibase Documentation
        5. JPA Documentation
        
        Args:
            question: The question to answer.
            category: The category of documentation to search in (all, jpa, liquibase, internal, examples, java).
        
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
            elif category == "java" and self._java_files_available:
                java_files = self._get_relevant_java_files(question)
                if java_files:
                    relevant_docs["java_files"] = java_files
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
        
        # 3. Third priority: Java Files (if available)
        if total_docs < min_docs_threshold and self._java_files_available and "java" in [category, "all"]:
            java_files = self._get_relevant_java_files(question)
            if java_files:
                relevant_docs["java_files"] = java_files
                total_docs += len(java_files)
        
        # 4. Fourth priority: Liquibase Documentation
        if total_docs < min_docs_threshold and "liquibase" in [category, "all"]:
            liquibase_docs = self._get_relevant_liquibase_docs(question)
            if liquibase_docs:
                relevant_docs["liquibase_docs"] = liquibase_docs
                total_docs += len(liquibase_docs)
        
        # 5. Fifth priority: JPA Documentation
        if total_docs < min_docs_threshold and "jpa" in [category, "all"]:
            jpa_docs = self._get_relevant_jpa_docs(question)
            if jpa_docs:
                relevant_docs["jpa_docs"] = jpa_docs
        
        return relevant_docs
    
    def _get_relevant_java_files(self, question: str) -> List[str]:
        """
        Get relevant Java files.
        
        Args:
            question: The question to answer.
        
        Returns:
            A list of relevant Java files.
        """
        if not self._java_files_available:
            return []
            
        # Get relevant documents
        docs = self.java_files_retriever.get_relevant_documents(question)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
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
        1. Example Migrations (YAML and XML) (highest priority)
        2. Java Files
        3. Internal Guidelines
        4. Liquibase Documentation
        5. JPA Documentation
        
        Args:
            relevant_docs: A dictionary of relevant documents by category.
        
        Returns:
            Combined context.
        """
        context_parts = []
        
        # 1. First priority: Example Migrations
        if "example_migrations" in relevant_docs and relevant_docs["example_migrations"]:
            context_parts.append("## Example Migrations (Highest Priority)\n\n" + "\n\n".join(relevant_docs["example_migrations"]))
        
        # 2. Second priority: Java Files
        if "java_files" in relevant_docs and relevant_docs["java_files"]:
            context_parts.append("## Java Files (High Priority)\n\n" + "\n\n".join(relevant_docs["java_files"]))
        
        # 3. Third priority: Internal Guidelines
        if "internal_guidelines" in relevant_docs and relevant_docs["internal_guidelines"]:
            context_parts.append("## Internal Guidelines (Medium Priority)\n\n" + "\n\n".join(relevant_docs["internal_guidelines"]))
        
        # 4. Fourth priority: Liquibase Documentation
        if "liquibase_docs" in relevant_docs and relevant_docs["liquibase_docs"]:
            context_parts.append("## Liquibase Documentation (Low Priority)\n\n" + "\n\n".join(relevant_docs["liquibase_docs"]))
        
        # 5. Fifth priority: JPA Documentation
        if "jpa_docs" in relevant_docs and relevant_docs["jpa_docs"]:
            context_parts.append("## JPA/Hibernate Documentation (Lowest Priority)\n\n" + "\n\n".join(relevant_docs["jpa_docs"]))
        
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
        1. Example Migrations (YAML and XML) (highest priority)
        2. Java Files (high priority)
        3. Internal Guidelines (medium priority)
        4. Liquibase Documentation (low priority)
        5. JPA Documentation (lowest priority)
        6. Your general knowledge (only if no other sources have relevant information)
        
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
