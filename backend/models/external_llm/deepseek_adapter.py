"""
DeepSeek LLM adapter for the Database Copilot.

This module provides an adapter for DeepSeek's API that can be used
with the Database Copilot application.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from backend.models.external_llm.base import BaseExternalLLM

logger = logging.getLogger(__name__)

class DeepSeekAdapter(BaseExternalLLM):
    """
    Adapter for DeepSeek's API.
    """
    
    provider_name: str = "deepseek"
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
        Initialize the DeepSeek adapter.
        
        Args:
            model_name: The name of the model to use. Defaults to "deepseek-chat".
            api_key: The API key to use. Defaults to the DEEPSEEK_API_KEY environment variable.
            temperature: The temperature to use for generation. Defaults to 0.2.
            max_tokens: The maximum number of tokens to generate. Defaults to 2048.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        # Get API key from environment variable if not provided
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DeepSeek API key is required. "
                "Please set the DEEPSEEK_API_KEY environment variable or pass it as an argument."
            )
        
        # Get model name from environment variable if not provided
        model_name = model_name or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            **kwargs
        )
        
        # Set DeepSeek-specific attributes
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Import OpenAI library (DeepSeek uses OpenAI-compatible API)
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",  # DeepSeek API endpoint
            )
        except ImportError:
            raise ImportError(
                "The openai package is required to use the DeepSeek adapter. "
                "Please install it with `pip install openai`."
            )
    
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> ChatResult:
        """
        Generate a response from DeepSeek based on the input messages.
        
        Args:
            messages: A list of messages to send to DeepSeek.
            stop: A list of strings to stop generation when encountered.
            **kwargs: Additional keyword arguments to pass to the DeepSeek API.
            
        Returns:
            A ChatResult containing the generated response.
        """
        # Extract system message and chat history
        system_message = self._extract_system_message(messages)
        chat_history = self._extract_chat_history(messages)
        
        # Prepare messages for DeepSeek API (which uses OpenAI-compatible format)
        deepseek_messages = []
        
        # Add system message if present
        if system_message:
            deepseek_messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Add chat history
        deepseek_messages.extend(chat_history)
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=deepseek_messages,
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
            logger.error(f"Error generating response from DeepSeek: {e}")
            raise
