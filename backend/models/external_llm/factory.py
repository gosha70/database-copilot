"""
Factory for creating external LLM instances.

This module provides a factory function for creating external LLM instances
based on the configured LLM type.
"""
import os
import logging
from typing import Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

def create_external_llm(llm_type: str) -> Optional[BaseChatModel]:
    """
    Create an external LLM instance based on the configured LLM type.
    
    Args:
        llm_type: The type of LLM to create. Must be one of "openai", "claude", "gemini", "mistral", or "deepseek".
        
    Returns:
        An initialized external LLM instance, or None if the LLM type is not supported.
    """
    llm_type = llm_type.lower()
    
    # Log all environment variables for debugging
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        if "API_KEY" in key:
            # Mask API keys for security
            logger.info(f"{key}: {'*' * 10}")
        else:
            logger.info(f"{key}: {value}")
    
    if llm_type == "openai":
        try:
            from backend.models.external_llm.openai_adapter import OpenAIAdapter
            # Log the API key (masked) for debugging
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                logger.info(f"Found OPENAI_API_KEY in environment: {'*' * min(10, len(api_key))}")
            else:
                logger.error("OPENAI_API_KEY not found in environment")
            
            return OpenAIAdapter()
        except (ImportError, ValueError) as e:
            logger.error(f"Failed to create OpenAI LLM: {e}")
            raise ValueError(f"Failed to create OpenAI LLM: {e}")
    
    elif llm_type == "claude":
        try:
            from backend.models.external_llm.claude_adapter import ClaudeAdapter
            # Log the API key (masked) for debugging
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                logger.info(f"Found ANTHROPIC_API_KEY in environment: {'*' * min(10, len(api_key))}")
            else:
                logger.error("ANTHROPIC_API_KEY not found in environment")
                
            return ClaudeAdapter()
        except (ImportError, ValueError) as e:
            logger.error(f"Failed to create Claude LLM: {e}")
            raise ValueError(f"Failed to create Claude LLM: {e}")
    
    elif llm_type == "gemini":
        try:
            from backend.models.external_llm.gemini_adapter import GeminiAdapter
            # Log the API key (masked) for debugging
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key:
                logger.info(f"Found GOOGLE_API_KEY in environment: {'*' * min(10, len(api_key))}")
            else:
                logger.error("GOOGLE_API_KEY not found in environment")
                
            return GeminiAdapter()
        except (ImportError, ValueError) as e:
            logger.error(f"Failed to create Gemini LLM: {e}")
            raise ValueError(f"Failed to create Gemini LLM: {e}")
    
    elif llm_type == "mistral":
        try:
            from backend.models.external_llm.mistral_adapter import MistralAdapter
            # Log the API key (masked) for debugging
            api_key = os.environ.get("MISTRAL_API_KEY")
            if api_key:
                logger.info(f"Found MISTRAL_API_KEY in environment: {'*' * min(10, len(api_key))}")
            else:
                logger.error("MISTRAL_API_KEY not found in environment")
                
            return MistralAdapter()
        except (ImportError, ValueError) as e:
            logger.error(f"Failed to create Mistral LLM: {e}")
            raise ValueError(f"Failed to create Mistral LLM: {e}")
    
    elif llm_type == "deepseek":
        try:
            from backend.models.external_llm.deepseek_adapter import DeepSeekAdapter
            # Log the API key (masked) for debugging
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if api_key:
                logger.info(f"Found DEEPSEEK_API_KEY in environment: {'*' * min(10, len(api_key))}")
            else:
                logger.error("DEEPSEEK_API_KEY not found in environment")
                
            return DeepSeekAdapter()
        except (ImportError, ValueError) as e:
            logger.error(f"Failed to create DeepSeek LLM: {e}")
            raise ValueError(f"Failed to create DeepSeek LLM: {e}")
    
    else:
        error_msg = f"Unsupported LLM type: {llm_type}. Supported types are: openai, claude, gemini, mistral, deepseek"
        logger.error(error_msg)
        raise ValueError(error_msg)
