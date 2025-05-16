from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import PrivateAttr
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

class OpenAIAdapter(BaseChatModel):
    provider_name: str = "openai"
    is_external_llm: bool = True

    _api_key: str = PrivateAttr()
    _model_name: str = PrivateAttr()
    _temperature: float = PrivateAttr()
    _max_tokens: int = PrivateAttr()
    _client: Any = PrivateAttr()

    @property
    def model_name(self):
        return self._model_name

    @property
    def api_key(self):
        return self._api_key

    @property
    def temperature(self):
        return self._temperature

    @property
    def max_tokens(self):
        return self._max_tokens

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        super().__init__()

        object.__setattr__(self, "_api_key", api_key or os.environ.get("OPENAI_API_KEY"))
        object.__setattr__(self, "_model_name", model_name)
        object.__setattr__(self, "_temperature", temperature)
        object.__setattr__(self, "_max_tokens", max_tokens)

        try:
            import openai
            object.__setattr__(self, "_client", openai.OpenAI(api_key=self._api_key))
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        
        if not self._api_key:
            raise ValueError("Missing OPENAI_API_KEY")

        logger.info(f"OpenAIAdapter initialized with model {self._model_name}")

    def _generate(self, messages, stop=None, **kwargs):
        openai_messages = [
            {"role": "system" if isinstance(m, SystemMessage) else
             "user" if isinstance(m, HumanMessage) else
             "assistant" if isinstance(m, AIMessage) else m.role,
             "content": m.content}
            for m in messages
        ]
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stop=stop,
            **kwargs
        )
        text = response.choices[0].message.content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @property
    def _llm_type(self) -> str:
        return f"external_{self.provider_name}"
