"""
Base class for external LLM adapters.

This module provides a base class for external LLM adapters that can be used
with the Database Copilot application.
"""
import logging
from typing import Any, Dict, List, Optional, Union, Mapping

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)

class BaseExternalLLM(BaseChatModel):
    """
    Base class for external LLM adapters.
    """
    # Class variables
    is_external_llm: bool = True
    """Flag to indicate this is an external LLM."""
    
    model_name: str
    """The name of the model to use."""
    
    provider_name: str
    """The name of the LLM provider."""
    
    def __init__(self, **kwargs):
        """
        Initialize the external LLM adapter.

        Args:
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        logger.info(f"USING EXTERNAL LLM: {self.provider_name} with model {self.model_name}")
    
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> ChatResult:
        """
        Generate a response from the LLM based on the input messages.
        
        Args:
            messages: A list of messages to send to the LLM.
            stop: A list of strings to stop generation when encountered.
            **kwargs: Additional keyword arguments to pass to the LLM.
            
        Returns:
            A ChatResult containing the generated response.
        """
        raise NotImplementedError("Subclasses must implement _generate method")
    
    def _extract_system_message(self, messages: List[BaseMessage]) -> Optional[str]:
        """
        Extract the system message from a list of messages.
        
        Args:
            messages: A list of messages to extract the system message from.
            
        Returns:
            The content of the system message, or None if no system message is found.
        """
        for message in messages:
            if isinstance(message, SystemMessage):
                return message.content
        return None
    
    def _extract_chat_history(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        Extract the chat history from a list of messages.
        
        Args:
            messages: A list of messages to extract the chat history from.
            
        Returns:
            A list of dictionaries containing the role and content of each message.
        """
        chat_history = []
        for message in messages:
            if isinstance(message, HumanMessage):
                chat_history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                chat_history.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                # Skip system messages as they are handled separately
                continue
            elif isinstance(message, ChatMessage):
                role = message.role
                if role == "system":
                    # Skip system messages as they are handled separately
                    continue
                chat_history.append({"role": role, "content": message.content})
        return chat_history
    
    def _create_chat_result(self, response_text: str) -> ChatResult:
        """
        Create a ChatResult from a response text.
        
        Args:
            response_text: The text response from the LLM.
            
        Returns:
            A ChatResult containing the response text.
        """
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        """
        Return the type of LLM.
        
        Returns:
            The type of LLM.
        """
        return f"external_{self.provider_name}"
