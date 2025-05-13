"""
Mistral LLM adapter for the Database Copilot.

This module provides an adapter for Mistral AI's API that can be used
with the Database Copilot application.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from backend.models.external_llm.base import BaseExternalLLM

logger = logging.getLogger(__name__)

class MistralAdapter(BaseExternalLLM):
    """
    Adapter for Mistral AI's API.
    """
    
    provider_name: str = "mistral"
    """The name of the LLM provider."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize the Mistral adapter.
        
        Args:
            model_name: The name of the model to use. Defaults to "mistral-medium".
            api_key: The API key to use. Defaults to the MISTRAL_API_KEY environment variable.
            temperature: The temperature to use for generation. Defaults to 0.2.
            max_tokens: The maximum number of tokens to generate. Defaults to 2048.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        # Get API key from environment variable if not provided
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "Mistral API key is required. "
                "Please set the MISTRAL_API_KEY environment variable or pass it as an argument."
            )
        
        # Get model name from environment variable if not provided
        model_name = model_name or os.environ.get("MISTRAL_MODEL", "mistral-medium")
        
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            **kwargs
        )
        
        # Set Mistral-specific attributes
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Import Mistral library
        try:
            import mistralai.client
            from mistralai.client import MistralClient
            self.client = MistralClient(api_key=api_key)
        except ImportError:
            raise ImportError(
                "The mistralai package is required to use the Mistral adapter. "
                "Please install it with `pip install mistralai`."
            )
    
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> ChatResult:
        """
        Generate a response from Mistral based on the input messages.
        
        Args:
            messages: A list of messages to send to Mistral.
            stop: A list of strings to stop generation when encountered.
            **kwargs: Additional keyword arguments to pass to the Mistral API.
            
        Returns:
            A ChatResult containing the generated response.
        """
        # Extract system message and chat history
        system_message = self._extract_system_message(messages)
        chat_history = self._extract_chat_history(messages)
        
        # Prepare messages for Mistral API
        mistral_messages = []
        
        # Add system message if present
        if system_message:
            mistral_messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Add chat history
        for message in chat_history:
            role = message["role"]
            content = message["content"]
            
            # Mistral uses "user" and "assistant" roles
            if role == "user":
                mistral_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                mistral_messages.append({
                    "role": "assistant",
                    "content": content
                })
        
        # Generate response
        try:
            # Import here to avoid circular imports
            from mistralai.models.chat_completion import ChatMessage
            
            # Convert dictionary messages to ChatMessage objects
            chat_messages = [
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in mistral_messages
            ]
            
            # Generate response
            response = self.client.chat(
                model=self.model_name,
                messages=chat_messages,
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
            logger.error(f"Error generating response from Mistral: {e}")
            raise
