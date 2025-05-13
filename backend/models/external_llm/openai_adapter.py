"""
OpenAI LLM adapter for the Database Copilot.

This module provides an adapter for OpenAI's API that can be used
with the Database Copilot application.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field, PrivateAttr

from backend.models.external_llm.base import BaseExternalLLM

logger = logging.getLogger(__name__)

class OpenAIAdapter(BaseExternalLLM):
    """
    Adapter for OpenAI's API.
    """
    
    provider_name: str = "openai"
    """The name of the LLM provider."""
    
    api_key: str = Field(None, description="OpenAI API key")
    temperature: float = Field(0.2, description="Temperature for generation")
    max_tokens: int = Field(2048, description="Maximum tokens to generate")
    
    # Private attributes that won't be part of the model schema
    _client = PrivateAttr()
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize the OpenAI adapter.
        
        Args:
            model_name: The name of the model to use. Defaults to "gpt-4o".
            api_key: The API key to use. Defaults to the OPENAI_API_KEY environment variable.
            temperature: The temperature to use for generation. Defaults to 0.2.
            max_tokens: The maximum number of tokens to generate. Defaults to 2048.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        # Get API key from environment variable if not provided
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Please set the OPENAI_API_KEY environment variable or pass it as an argument."
            )
        
        # Get model name from environment variable if not provided
        model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o")
        
        # Initialize parent class with all parameters
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Import OpenAI library
        try:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Successfully initialized OpenAI client with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to import OpenAI library: {e}")
            raise ImportError(
                f"Error initializing OpenAI client: {e}. "
                "The openai package is required to use the OpenAI adapter. "
                "Please install it with `pip install openai`."
            )
    
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> ChatResult:
        """
        Generate a response from OpenAI based on the input messages.
        
        Args:
            messages: A list of messages to send to OpenAI.
            stop: A list of strings to stop generation when encountered.
            **kwargs: Additional keyword arguments to pass to the OpenAI API.
            
        Returns:
            A ChatResult containing the generated response.
        """
        # Extract system message and chat history
        system_message = self._extract_system_message(messages)
        chat_history = self._extract_chat_history(messages)
        
        # Prepare messages for OpenAI API
        openai_messages = []
        
        # Add system message if present
        if system_message:
            openai_messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Add chat history
        openai_messages.extend(chat_history)
        
        # Generate response
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
                **kwargs
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Create and return ChatResult
            return self._create_chat_result(response_text)
        
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            raise
