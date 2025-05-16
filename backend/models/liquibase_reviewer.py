"""
Liquibase migration reviewer using RAG.
"""
import logging
from typing import Dict, List, Optional, Union, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import NUM_RETRIEVAL_RESULTS
from backend.models.llm import get_llm
from backend.models.vector_store import get_retriever
from backend.models.liquibase_parser import LiquibaseParser

logger = logging.getLogger(__name__)

class LiquibaseReviewer:
    """
    Reviewer for Liquibase migrations using RAG.
    """
    
    def __init__(self, debug_mode=False):
        """
        Initialize the reviewer.
        
        Args:
            debug_mode: Whether to enable debug mode, which logs more information.
        """
        self.parser = LiquibaseParser()
        self.debug_mode = debug_mode
        
        # Import at the module level to avoid circular imports
        from backend.models.streamlit_compatibility import get_safe_llm
        from backend.config import get_current_llm_type
        
        # Use get_safe_llm instead of get_llm to avoid dependency issues
        try:
            # Try to use an external LLM if configured
            current_llm_type = get_current_llm_type()
            if current_llm_type != "local":
                self.llm = get_safe_llm(use_external=True)
            else:
                self.llm = get_safe_llm()
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise e
        
        # Get retrievers for different document categories
        self.liquibase_docs_retriever = get_retriever(collection_name="liquibase_docs")
        self.internal_guidelines_retriever = get_retriever(collection_name="internal_guidelines")
        self.example_migrations_retriever = get_retriever(collection_name="example_migrations")
        
        if self.debug_mode:
            logger.info("Debug mode enabled for LiquibaseReviewer")
    
    def review_migration(self, migration_content: str, format_type: str = "xml") -> str:
        """
        Review a Liquibase migration.
        
        Args:
            migration_content: The content of the migration file.
            format_type: The format of the migration file (xml or yaml).
        
        Returns:
            A review of the migration.
        """
        logger.info(f"Reviewing {format_type} migration")
        
        try:
            # Log which LLM is being used
            if hasattr(self.llm, 'is_external_llm') and self.llm.is_external_llm:
                logger.info(f"Using external LLM: {self.llm.provider_name} with model {self.llm.model_name}")
            else:
                logger.info(f"Using local LLM: {getattr(self.llm, '_llm_type', 'unknown')}")
            
            # Parse the migration
            parsed_migration = self._parse_migration_content(migration_content, format_type)
            if not parsed_migration or (isinstance(parsed_migration, dict) and "error" in parsed_migration):
                error_msg = parsed_migration.get("error", "Unknown parsing error") if isinstance(parsed_migration, dict) else "Failed to parse migration"
                logger.error(f"Error parsing migration: {error_msg}")
                return f"Error: Failed to parse the migration file. {error_msg}"
            
            # Get relevant documents from different categories
            try:
                liquibase_docs = self._get_relevant_liquibase_docs(parsed_migration)
                logger.info(f"Retrieved {len(liquibase_docs)} Liquibase docs")
            except Exception as e:
                logger.error(f"Error retrieving Liquibase docs: {e}")
                liquibase_docs = []
            
            try:
                internal_guidelines = self._get_relevant_internal_guidelines(parsed_migration)
                logger.info(f"Retrieved {len(internal_guidelines)} internal guidelines")
            except Exception as e:
                logger.error(f"Error retrieving internal guidelines: {e}")
                internal_guidelines = []
            
            try:
                example_migrations = self._get_relevant_example_migrations(parsed_migration)
                logger.info(f"Retrieved {len(example_migrations)} example migrations")
            except Exception as e:
                logger.error(f"Error retrieving example migrations: {e}")
                example_migrations = []
            
            # Check if we have any context
            if not liquibase_docs and not internal_guidelines and not example_migrations:
                logger.warning("No context retrieved from any source")
                # Continue with empty context, but log a warning
            
            # Combine all relevant documents
            context = self._combine_context(liquibase_docs, internal_guidelines, example_migrations)
            
            # Log the context if debug mode is enabled
            if self.debug_mode:
                logger.info("=== DEBUG: RETRIEVED CONTEXT ===")
                logger.info(f"Context length: {len(context)} characters")
                logger.info("=== Liquibase Docs ===")
                for i, doc in enumerate(liquibase_docs):
                    logger.info(f"Doc {i+1} (length: {len(doc)} chars): {doc[:200]}...")
                logger.info("=== Internal Guidelines ===")
                for i, doc in enumerate(internal_guidelines):
                    logger.info(f"Doc {i+1} (length: {len(doc)} chars): {doc[:200]}...")
                logger.info("=== Example Migrations ===")
                for i, doc in enumerate(example_migrations):
                    logger.info(f"Doc {i+1} (length: {len(doc)} chars): {doc[:200]}...")
                logger.info("=== END DEBUG: RETRIEVED CONTEXT ===")
            
            # Generate a basic review template
            review_template = self._generate_review_template(parsed_migration, format_type)
            logger.info(f"Generated review template: {review_template[:200]}...")
            
            # Create the prompt template with the review template
            prompt = ChatPromptTemplate.from_template("""
You are an expert Liquibase and JPA reviewer. 
Your task is to enhance the provided review template with detailed analysis.

# Review Template (BASE YOUR RESPONSE ON THIS)
{review_template}

# Migration to Review
```{format_type}
{migration_content}
```

# Parsed Migration Structure
```
{parsed_migration}
```

IMPORTANT INSTRUCTIONS:
1. Start with the review template above and enhance it. DO NOT create a completely different review.
2. Keep the same structure and sections from the template.
3. Fill in the checklist with ✅ or ❌ for each item based on your analysis.
4. Add specific issues, recommendations, and best practices based on the migration.
5. Be specific and reference actual code from the migration.
6. If you can't determine if a checklist item applies, use ✅ but note the uncertainty.

Return the complete review with all sections filled in.
""")
            
            # Log prompt length and content for debugging
            prompt_content = prompt.format(
                review_template=review_template,
                migration_content=migration_content,
                format_type=format_type,
                parsed_migration=str(parsed_migration)
            )
            logger.info(f"LLM prompt length: {len(prompt_content)} characters")
            logger.debug(f"LLM prompt content:\n{prompt_content}")

            review_chain = prompt | self.llm | StrOutputParser()

            # Generate the review
            try:
                # Set up a synchronous environment for the LLM call
                import asyncio

                # Check if we're in an event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in a running event loop, use it
                        logger.info("Using existing event loop")
                    else:
                        # Loop exists but isn't running, create a new one
                        logger.info("Creating new event loop")
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                except RuntimeError:
                    # No event loop exists, create a new one
                    logger.info("No event loop found, creating new one")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Run the LLM call
                review = review_chain.invoke({
                    "review_template": review_template,
                    "migration_content": migration_content,
                    "format_type": format_type,
                    "parsed_migration": str(parsed_migration)
                })

                logger.info(f"Raw LLM review output: {repr(review)}")
                
                # Check if the review contains error messages
                if review and (review.startswith("ERROR:") or "This is a placeholder response from a fallback system" in review):
                    logger.error(f"LLM returned an error: {review}")
                    return f"Error: The LLM returned an error response. Please check the logs for details."
                
                # Post-process the review to remove any template placeholders
                review = self._post_process_review(review)
                logger.info(f"Post-processed review output: {repr(review)}")
                
                # If the review is empty or too short after post-processing, fall back to the template
                if not review or len(review) < 100:
                    logger.warning("Review is too short after post-processing, falling back to template")
                    return review_template
                
                return review
                
            except Exception as e:
                logger.error(f"Error generating review: {e}")
                # Fall back to the template if there's an error
                logger.info("Falling back to review template due to error")
                return review_template
            
        except Exception as e:
            logger.error(f"Error in review_migration: {e}")
            return f"Error: An unexpected error occurred during the review process. {str(e)}"
    
    def _generate_review_template(self, parsed_migration: Dict[str, Any], format_type: str) -> str:
        """
        Generate a basic review template from the parsed migration.
        
        Args:
            parsed_migration: The parsed migration.
            format_type: The format of the migration file (xml or yaml).
            
        Returns:
            A basic review template.
        """
        # Extract information from the parsed migration
        table_names = self._extract_table_names(parsed_migration)
        change_types = self._extract_change_types(parsed_migration)
        
        # Create a summary based on the change types and table names
        summary = f"This migration performs {', '.join(change_types)} operations on {', '.join(table_names)} table(s)."
        
        # Create the template
        template_lines = []
        
        # Add LLM info
        if hasattr(self.llm, 'is_external_llm') and self.llm.is_external_llm:
            llm_info = f"🤖 Using external LLM: {self.llm.provider_name} with model {self.llm.model_name}"
        else:
            llm_info = f"🤖 Using local LLM: {getattr(self.llm, '_llm_type', 'unknown')}"
        template_lines.append(llm_info)
        template_lines.append("")
        
        # Add summary
        template_lines.append("## Summary 📄")
        template_lines.append(summary)
        template_lines.append("")
        
        # Add checklist
        template_lines.append("# Liquibase / RDBMS Code Review Checklist ✅❌")
        template_lines.append("")
        template_lines.append("- [ ] Migration id consistent across all affected versions")
        template_lines.append("- [ ] All identifier names <=30 characters")
        template_lines.append("- [ ] All identifier names are lowercase_underscore_form")
        template_lines.append("- [ ] Abbreviations are standard and/or sensible")
        template_lines.append("- [ ] Table names are singular, Join Tables are plural")
        template_lines.append("- [ ] Table names not repeated in column names")
        template_lines.append("- [ ] Column type is not repeated in column name")
        template_lines.append("- [ ] id is simply id, uuid is uuid")
        template_lines.append("- [ ] Constraints on columns ideally in form `<table-name>_<column-name>_<suffix>`")
        template_lines.append("- [ ] Foreign key suffix is _fk, Unique constraint is _uc, Index is _idx")
        template_lines.append("- [ ] All FKs also have an index added if no unique constraint present")
        template_lines.append("- [ ] All constraints named")
        template_lines.append("- [ ] Every auto-increment id column has a sequence `<table-name>_sq` for Oracle")
        template_lines.append("- [ ] No preconditions avoiding index creation")
        template_lines.append("- [ ] Unit tests present if non-trivial logic")
        template_lines.append("- [ ] Cascading behavior specified in Liquibase and JPA")
        template_lines.append("- [ ] JPA `@Column` `nullable` configuration matches Liquibase")
        template_lines.append("- [ ] All JPA string `@Column`s set `length`")
        template_lines.append("- [ ] JPA names use camelCase")
        template_lines.append("- [ ] If `FetchType.LAZY` is used for lists, there was an effort to avoid LazyInitializationException")
        template_lines.append("- [ ] Implement `toString` on Entities, excluding lazily-loaded data")
        template_lines.append("")
        
        # Add issues section
        template_lines.append("## Issues 🐞")
        template_lines.append("No issues identified yet.")
        template_lines.append("")
        
        # Add recommendations section
        template_lines.append("## Recommendations 💡")
        template_lines.append("No recommendations yet.")
        template_lines.append("")
        
        # Add best practices section
        template_lines.append("## Best Practices 🌟")
        template_lines.append("The migration follows standard Liquibase practices.")
        
        return "\n".join(template_lines)
    
    def _post_process_review(self, review: str) -> str:
        """
        Post-process the generated review to remove any unwanted text.
        
        Args:
            review: The generated review.
        
        Returns:
            The post-processed review.
        """
        # Check if the review contains the DO NOT instructions that should have been removed
        if "DO NOT include" in review or "DO NOT RETURN" in review:
            logger.warning("Review contains prompt instructions that should have been removed")
            # Clean up the review by removing the DO NOT instructions
            lines = review.split('\n')
            cleaned_lines = [line for line in lines if not (line.strip().startswith("DO NOT") or "DO NOT" in line)]
            review = '\n'.join(cleaned_lines)
        
        # Check if the review contains template text
        template_indicators = [
            "You may use the following sections as a template",
            "Remember, the goal is to provide",
            "Your review should be",
            "the following sections as a template",
            "sections as a template",
            "template for your review"
        ]
        
        for indicator in template_indicators:
            if indicator.lower() in review.lower():
                logger.warning(f"Review contains template text: {indicator}")
                # Return a clear error message
                return """
# Error: LLM returned template instructions instead of an actual review

The LLM failed to generate a proper review and instead returned template instructions.

This is likely due to one of the following issues:
1. The LLM model is not powerful enough to handle the task
2. The context window is too small for the model to process all the information
3. The model needs more specific examples to understand the task

Please try one of the following solutions:
1. Use an external LLM like OpenAI's GPT-4 or Anthropic's Claude (see external_llm_instructions.md)
2. Provide a simpler migration file to review
3. Check that the vector store has been properly built with relevant examples
"""
        
        # Remove any lines that contain template placeholders
        lines = review.split('\n')
        cleaned_lines = []
        skip_section = False
        
        for line in lines:
            # Skip lines that contain template placeholders
            if any(phrase in line.lower() for phrase in ["[list specific", "[provide", "[evaluate", "[highlight", "following sections", "template", "remember,"]):
                skip_section = True
                continue
            
            # Skip lines that are just placeholders in square brackets
            if line.strip().startswith("[") and line.strip().endswith("]"):
                continue
            
            # Reset skip_section flag on new section headers
            if line.strip().startswith("##"):
                skip_section = False
            
            # Skip empty lines after template text
            if skip_section and not line.strip():
                continue
            
            # Add the line to the cleaned lines if not skipping
            if not skip_section:
                cleaned_lines.append(line)
        
        # Join the cleaned lines
        cleaned_review = '\n'.join(cleaned_lines)
        
        # Remove any leading/trailing whitespace
        cleaned_review = cleaned_review.strip()
        
        # If the review is empty after cleaning, return an error message
        if not cleaned_review:
            logger.error("Review is empty after post-processing")
            return """
# Error: Failed to generate a meaningful review

The LLM failed to generate a proper review. Please try one of the following solutions:
1. Use an external LLM like OpenAI's GPT-4 or Anthropic's Claude (see external_llm_instructions.md)
2. Provide a simpler migration file to review
3. Check that the vector store has been properly built with relevant examples
"""
        
        return cleaned_review
    
    def _parse_migration_content(self, migration_content: str, format_type: str) -> Dict[str, Any]:
        """
        Parse migration content.
        
        Args:
            migration_content: The content of the migration file.
            format_type: The format of the migration file (xml or yaml).
        
        Returns:
            A dictionary containing the parsed migration.
        """
        # Create a temporary file to parse
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=f".{format_type}", delete=False) as temp_file:
            temp_file.write(migration_content.encode())
            temp_file_path = temp_file.name
        
        try:
            # Parse the file
            parsed_migration = self.parser.parse_file(temp_file_path)
            return parsed_migration
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    
    def _get_relevant_liquibase_docs(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Get relevant Liquibase documentation.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of relevant Liquibase documentation.
        """
        # Extract change types from the parsed migration
        change_types = self._extract_change_types(parsed_migration)
        
        # Create a query based on the change types
        query = f"Liquibase documentation for: {', '.join(change_types)}"
        
        # Get relevant documents
        docs = self.liquibase_docs_retriever.invoke(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _get_relevant_internal_guidelines(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Get relevant internal guidelines.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of relevant internal guidelines.
        """
        # Extract table names and change types from the parsed migration
        table_names = self._extract_table_names(parsed_migration)
        change_types = self._extract_change_types(parsed_migration)
        
        # Create a query based on the table names and change types
        query = f"Internal guidelines for database migrations with: {', '.join(change_types)} on tables: {', '.join(table_names)}"
        
        # Get relevant documents
        docs = self.internal_guidelines_retriever.invoke(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _get_relevant_example_migrations(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Get relevant example migrations.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of relevant example migrations.
        """
        # Extract table names and change types from the parsed migration
        table_names = self._extract_table_names(parsed_migration)
        change_types = self._extract_change_types(parsed_migration)
        
        # Create a query based on the table names and change types
        query = f"Example migrations with: {', '.join(change_types)} on tables: {', '.join(table_names)}"
        
        # Get relevant documents
        docs = self.example_migrations_retriever.invoke(query)
        
        # Extract the content from the documents
        return [doc.page_content for doc in docs]
    
    def _extract_change_types(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Extract change types from a parsed migration.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of change types.
        """
        change_types = set()
        
        # Extract change types from changesets
        if 'databaseChangeLog' in parsed_migration:
            # Handle both list and dict formats for databaseChangeLog
            changeset_list = []
            if isinstance(parsed_migration['databaseChangeLog'], dict):
                changeset_list = parsed_migration['databaseChangeLog'].get('changeSet', [])
            elif isinstance(parsed_migration['databaseChangeLog'], list):
                # If databaseChangeLog is a list, find the changeSet entries
                for item in parsed_migration['databaseChangeLog']:
                    if isinstance(item, dict) and 'changeSet' in item:
                        if isinstance(item['changeSet'], list):
                            changeset_list.extend(item['changeSet'])
                        else:
                            changeset_list.append(item['changeSet'])
            
            # Process each changeset
            for changeset in changeset_list:
                # Handle both list and dict formats for changes
                changes_list = []
                if isinstance(changeset, dict) and 'changes' in changeset:
                    changes = changeset['changes']
                    if isinstance(changes, list):
                        changes_list = changes
                    elif isinstance(changes, dict):
                        changes_list = [changes]
                
                # Extract change types
                for change in changes_list:
                    if isinstance(change, dict):
                        change_types.update(change.keys())
        
        return list(change_types) if change_types else ["unknown"]
    
    def _extract_table_names(self, parsed_migration: Dict[str, Any]) -> List[str]:
        """
        Extract table names from a parsed migration.
        
        Args:
            parsed_migration: The parsed migration.
        
        Returns:
            A list of table names.
        """
        table_names = set()
        
        # Extract table names from changesets
        if 'databaseChangeLog' in parsed_migration:
            # Handle both list and dict formats for databaseChangeLog
            changeset_list = []
            if isinstance(parsed_migration['databaseChangeLog'], dict):
                changeset_list = parsed_migration['databaseChangeLog'].get('changeSet', [])
            elif isinstance(parsed_migration['databaseChangeLog'], list):
                # If databaseChangeLog is a list, find the changeSet entries
                for item in parsed_migration['databaseChangeLog']:
                    if isinstance(item, dict) and 'changeSet' in item:
                        if isinstance(item['changeSet'], list):
                            changeset_list.extend(item['changeSet'])
                        else:
                            changeset_list.append(item['changeSet'])
            
            # Process each changeset
            for changeset in changeset_list:
                # Handle both list and dict formats for changes
                changes_list = []
                if isinstance(changeset, dict) and 'changes' in changeset:
                    changes = changeset['changes']
                    if isinstance(changes, list):
                        changes_list = changes
                    elif isinstance(changes, dict):
                        changes_list = [changes]
                
                # Extract table names from each change
                for change in changes_list:
                    if not isinstance(change, dict):
                        continue
                    
                    for change_type, change_data in change.items():
                        if not isinstance(change_data, dict):
                            continue
                            
                        # Extract table name based on change type
                        if change_type == 'createTable' and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
                        elif change_type == 'addColumn' and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
                        elif change_type == 'dropTable' and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
                        elif change_type == 'addForeignKeyConstraint':
                            if 'baseTableName' in change_data:
                                table_names.add(change_data['baseTableName'])
                            if 'referencedTableName' in change_data:
                                table_names.add(change_data['referencedTableName'])
                        elif change_type == 'addCheckConstraint' and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
                        elif change_type in ['addPrimaryKey', 'addUniqueConstraint', 'createIndex'] and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
                        elif change_type == 'update' and 'tableName' in change_data:
                            table_names.add(change_data['tableName'])
        
        return list(table_names) if table_names else ["unknown"]
    
    def _combine_context(
        self,
        liquibase_docs: List[str],
        internal_guidelines: List[str],
        example_migrations: List[str]
    ) -> str:
        """
        Combine context from different sources.
        
        Args:
            liquibase_docs: Relevant Liquibase documentation.
            internal_guidelines: Relevant internal guidelines.
            example_migrations: Relevant example migrations.
        
        Returns:
            Combined context.
        """
        context_parts = []
        
        # Add Liquibase documentation
        if liquibase_docs:
            context_parts.append("## Liquibase Documentation\n\n" + "\n\n".join(liquibase_docs))
        
        # Add internal guidelines
        if internal_guidelines:
            context_parts.append("## Internal Guidelines\n\n" + "\n\n".join(internal_guidelines))
        
        # Add example migrations
        if example_migrations:
            context_parts.append("## Example Migrations\n\n" + "\n\n".join(example_migrations))
        
        #print(f"=== DEBUG: Combined Context ===\n\n{context_parts}")
        return "\n\n".join(context_parts)
    
    def _create_review_chain(self):
        """
        Create a chain for reviewing migrations.
        
        Returns:
            A chain for reviewing migrations.
        """
        # Create the prompt template with detailed requirements
        prompt = ChatPromptTemplate.from_template("""
You are an expert Liquibase and JPA reviewer. 
Your task is to enhance the provided review template with detailed analysis.

# Review Template (BASE YOUR RESPONSE ON THIS)
{review_template}

# Migration to Review
```{format_type}
{migration_content}
```

# Parsed Migration Structure
```
{parsed_migration}
```

IMPORTANT INSTRUCTIONS:
1. Start with the review template above and enhance it. DO NOT create a completely different review.
2. Keep the same structure and sections from the template.
3. Fill in the checklist with ✅ or ❌ for each item based on your analysis.
4. Add specific issues, recommendations, and best practices based on the migration.
5. Be specific and reference actual code from the migration.
6. If you can't determine if a checklist item applies, use ✅ but note the uncertainty.

Return the complete review with all sections filled in.
""")

        
        # Create the chain
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
