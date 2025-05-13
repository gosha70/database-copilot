"""
Claude LLM adapter for the Database Copilot.

This module provides an adapter for Anthropic's Claude API that can be used
with the Database Copilot application.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

from backend.models.external_llm.base import BaseExternalLLM

logger = logging.getLogger(__name__)

class ClaudeAdapter(BaseExternalLLM):
    """
    Adapter for Anthropic's Claude API.
    """
    
    provider_name: str = "claude"
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
        Initialize the Claude adapter.
        
        Args:
            model_name: The name of the model to use. Defaults to "claude-3-opus-20240229".
            api_key: The API key to use. Defaults to the ANTHROPIC_API_KEY environment variable.
            temperature: The temperature to use for generation. Defaults to 0.2.
            max_tokens: The maximum number of tokens to generate. Defaults to 2048.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        # Get API key from environment variable if not provided
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Please set the ANTHROPIC_API_KEY environment variable or pass it as an argument."
            )
        
        # Get model name from environment variable if not provided
        model_name = model_name or os.environ.get("CLAUDE_MODEL", "claude-3-opus-20240229")
        
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            **kwargs
        )
        
        # Set Claude-specific attributes
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Import Anthropic library
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "The anthropic package is required to use the Claude adapter. "
                "Please install it with `pip install anthropic`."
            )
    
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> ChatResult:
        """
        Generate a response from Claude based on the input messages.
        
        Args:
            messages: A list of messages to send to Claude.
            stop: A list of strings to stop generation when encountered.
            **kwargs: Additional keyword arguments to pass to the Claude API.
            
        Returns:
            A ChatResult containing the generated response.
        """
        # Extract system message and chat history
        system_message = self._extract_system_message(messages)
        chat_history = self._extract_chat_history(messages)
        
        # Prepare messages for Claude API
        claude_messages = []
        
        # Add system message if present
        if system_message:
            claude_messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Add chat history
        for message in chat_history:
            # Claude uses "user" and "assistant" roles
            role = message["role"]
            if role == "user":
                claude_messages.append({
                    "role": "user",
                    "content": message["content"]
                })
            elif role == "assistant":
                claude_messages.append({
                    "role": "assistant",
                    "content": message["content"]
                })
        
        # Generate response
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=claude_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop_sequences=stop if stop else None,
                **kwargs
            )
            
            # Extract response text
            response_text = response.content[0].text
            
            # Create and return ChatResult
            return self._create_chat_result(response_text)
        
        except Exception as e:
            logger.error(f"Error generating response from Claude: {e}")
            raise
