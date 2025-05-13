"""
Gemini LLM adapter for the Database Copilot.

This module provides an adapter for Google's Gemini API that can be used
with the Database Copilot application.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from backend.models.external_llm.base import BaseExternalLLM

logger = logging.getLogger(__name__)

class GeminiAdapter(BaseExternalLLM):
    """
    Adapter for Google's Gemini API.
    """
    
    provider_name: str = "gemini"
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
        Initialize the Gemini adapter.
        
        Args:
            model_name: The name of the model to use. Defaults to "gemini-1.5-pro".
            api_key: The API key to use. Defaults to the GOOGLE_API_KEY environment variable.
            temperature: The temperature to use for generation. Defaults to 0.2.
            max_tokens: The maximum number of tokens to generate. Defaults to 2048.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        # Get API key from environment variable if not provided
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. "
                "Please set the GOOGLE_API_KEY environment variable or pass it as an argument."
            )
        
        # Get model name from environment variable if not provided
        model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
        
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            **kwargs
        )
        
        # Set Gemini-specific attributes
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Import Google Generative AI library
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai
            self.model = genai.GenerativeModel(model_name=model_name)
        except ImportError:
            raise ImportError(
                "The google-generativeai package is required to use the Gemini adapter. "
                "Please install it with `pip install google-generativeai`."
            )
    
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> ChatResult:
        """
        Generate a response from Gemini based on the input messages.
        
        Args:
            messages: A list of messages to send to Gemini.
            stop: A list of strings to stop generation when encountered.
            **kwargs: Additional keyword arguments to pass to the Gemini API.
            
        Returns:
            A ChatResult containing the generated response.
        """
        # Extract system message and chat history
        system_message = self._extract_system_message(messages)
        chat_history = self._extract_chat_history(messages)
        
        # Prepare messages for Gemini API
        gemini_messages = []
        
        # Add system message if present
        if system_message:
            # Gemini doesn't have a system message, so we'll add it as a user message
            # with a special prefix
            gemini_messages.append({
                "role": "user",
                "parts": [f"System: {system_message}"]
            })
            
            # Add a dummy assistant response to acknowledge the system message
            gemini_messages.append({
                "role": "model",
                "parts": ["I'll follow those instructions."]
            })
        
        # Add chat history
        for message in chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [content]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [content]
                })
        
        # Generate response
        try:
            # Create a chat session
            chat = self.model.start_chat(history=gemini_messages[:-1] if gemini_messages else [])
            
            # Get the last user message
            last_message = None
            for message in reversed(gemini_messages):
                if message["role"] == "user":
                    last_message = message["parts"][0]
                    break
            
            if not last_message:
                last_message = "Please provide a response."
            
            # Generate response
            response = chat.send_message(
                last_message,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "stop_sequences": stop if stop else None,
                }
            )
            
            # Extract response text
            response_text = response.text
            
            # Create and return ChatResult
            return self._create_chat_result(response_text)
        
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            raise
