"""
OpenAI LLM adapter for the Database Copilot.

This module provides an adapter for OpenAI's API that can be used
with the Database Copilot application.
"""
import os
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration

logger = logging.getLogger(__name__)

class OpenAIAdapter(BaseChatModel):
    """
    Adapter for OpenAI's API.
    """
    
    provider_name: str = "openai"
    is_external_llm: bool = True
    model_name: str
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Initialize the OpenAI adapter.
        
        Args:
            api_key: The API key to use. Defaults to the OPENAI_API_KEY environment variable.
            model_name: The name of the model to use. Defaults to "gpt-4o".
            temperature: The temperature to use for generation. Defaults to 0.2.
            max_tokens: The maximum number of tokens to generate. Defaults to 2048.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        # Get API key from environment variable if not provided
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        logger.info(f"API key from environment: {'*****' if api_key else 'None'}")
        
        # Check if streamlit is available and try to get the API key from secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                logger.info("Found OPENAI_API_KEY in streamlit secrets")
                api_key = api_key or st.secrets['OPENAI_API_KEY']
        except ImportError:
            logger.info("Streamlit not available, skipping secrets check")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Please set the OPENAI_API_KEY environment variable or in .streamlit/secrets.toml"
            )
        
        # Store attributes
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the OpenAI client
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Successfully initialized OpenAI client with model {model_name}")
        except ImportError:
            logger.error("Failed to import OpenAI library")
            raise ImportError(
                "The openai package is required to use the OpenAI adapter. "
                "Please install it with `pip install openai`."
            )
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise ValueError(
                f"Error initializing OpenAI client: {e}. "
                "Please check your API key and configuration."
            )
        
        # Initialize parent class
        super().__init__(**kwargs)
        logger.info(f"USING EXTERNAL LLM: {self.provider_name} with model {self.model_name}")
    
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
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                openai_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                openai_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                openai_messages.append({"role": "assistant", "content": message.content})
            elif hasattr(message, "role"):
                openai_messages.append({"role": message.role, "content": message.content})
        
        try:
            # Call the OpenAI API directly
            response = self.client.chat.completions.create(
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
            message = AIMessage(content=response_text)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        """
        Return the type of LLM.
        
        Returns:
            The type of LLM.
        """
        return f"external_{self.provider_name}"
