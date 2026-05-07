import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union
import requests

from loguru import logger
from openai import OpenAI, AsyncOpenAI

from ..core.model_base import ModelAdapterBase
from ..core.model_response import ModelResponse
from ..core.message import Message


class DeepSeekChatAdapter(ModelAdapterBase):
    """
    Adapter for DeepSeek Chat API (OpenAI-compatible), providing both sync and async methods.
    """

    def __init__(
        self,
        config_name: str,
        model_name: str,
        api_key: str,
        client_args: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        generate_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the DeepSeekChatAdapter.

        Args:
            config_name: Unique adapter configuration name.
            model_name: Model identifier (e.g., "deepseek-chat").
            api_key: Your DeepSeek API key.
            client_args: Additional parameters for OpenAI client (e.g., timeout).
            stream: Whether to request streaming responses (ignored if unsupported).
            generate_args: Default generation parameters (temperature, max_tokens, etc.).
            **kwargs: Extra arguments passed to base class.
        """
        super().__init__(config_name=config_name, model_name=model_name, **kwargs)
        self.stream = stream
        self.generate_args = generate_args or {}

        client_args = client_args or {}
        # Ensure base_url points to DeepSeek API
        default_base_url = "https://api.deepseek.com/v1/"
        client_args.setdefault("base_url", default_base_url)

        # store for list_models
        self._api_key = api_key
        self._base_url = client_args["base_url"].rstrip('/')

        # synchronous OpenAI-compatible client
        self.client = OpenAI(api_key=api_key, **client_args)

        # asynchronous client if available
        try:
            self.async_client = AsyncOpenAI(api_key=api_key, **client_args)
        except ImportError:
            logger.warning("AsyncOpenAI not available; acall will use threadpool.")
            self.async_client = None

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Synchronous chat completion using DeepSeek.

        Args:
            messages: List of dicts with 'role' and 'content'.
            stream: Streaming flag (may be ignored by DeepSeek).
            **kwargs: Override default generate_args.

        Returns:
            ModelResponse containing generated text and metadata.
        """
        call_kwargs = {**self.generate_args, **kwargs}
        use_stream = self.stream if stream is None else stream
        call_kwargs.update({
            "model": self.model_name,
            "messages": messages,
            "stream": use_stream
        })
        self._validate_messages(messages)

        try:
            resp = self.client.chat.completions.create(**call_kwargs)
            content = resp.choices[0].message.content
            return ModelResponse(
                text=content,
                raw=resp.model_dump(),
                usage=getattr(resp, "usage", None),
                model_info={"config_name": self.config_name, "model_name": self.model_name}
            )
        except Exception as e:
            logger.error(f"DeepSeek sync call failed: {e}")
            raise

    async def acall(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Asynchronous chat completion via AsyncOpenAI or threadpool fallback.
        """
        call_kwargs = {**self.generate_args, **kwargs}
        use_stream = self.stream if stream is None else stream
        call_kwargs.update({
            "model": self.model_name,
            "messages": messages,
            "stream": use_stream
        })
        self._validate_messages(messages)

        try:
            if self.async_client:
                resp = await self.async_client.chat.completions.create(**call_kwargs)
            else:
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(**call_kwargs)
                )
            content = resp.choices[0].message.content
            return ModelResponse(
                text=content,
                raw=resp.model_dump(),
                usage=getattr(resp, "usage", None),
                model_info={"config_name": self.config_name, "model_name": self.model_name}
            )
        except Exception as e:
            logger.error(f"DeepSeek async call failed: {e}")
            raise

    def list_models(self) -> List[str]:
        """
        Fetch available model IDs from DeepSeek /models endpoint.
        """
        url = f"{self._base_url}/models"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [item["id"] for item in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to list DeepSeek models: {e}")
            raise

    async def alist_models(self) -> List[str]:
        """
        Async version of list_models via threadpool.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.list_models)

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> None:
        """
        Ensure messages is a list of dicts containing 'role' and 'content'.
        """
        if not isinstance(messages, list):
            raise ValueError(f"Messages must be a list, got {type(messages)}")
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Each message must be a dict with 'role' and 'content'")

    def format(self, *args: Union[Message, Sequence[Message]]) -> List[Dict[str, Any]]:
        """
        Convert Message objects into standard chat dicts for DeepSeek.
        """
        return ModelAdapterBase.format_for_common_chat_models(*args)


class DeepSeekEmbeddingAdapter(ModelAdapterBase):
    """
    Adapter for DeepSeek Embedding (not available).

    DeepSeek does not currently support an embeddings API. You may use
    their Chat API to simulate embeddings by prompting for vector output,
    or use a separate embedding service.
    """

    def __init__(
        self,
        config_name: str,
        model_name: str,
        api_key: str,
        client_args: Optional[Dict[str, Any]] = None,
        generate_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(config_name=config_name, model_name=model_name, **kwargs)
        raise NotImplementedError("DeepSeek API does not support embeddings.")

    def __call__(self, *args, **kwargs) -> ModelResponse:
        raise NotImplementedError("DeepSeek embeddings not supported.")

    async def acall(self, *args, **kwargs) -> ModelResponse:
        raise NotImplementedError("DeepSeek embeddings not supported.")

    def list_models(self) -> List[str]:
        raise NotImplementedError("DeepSeek embeddings not supported.")

    def format(self, *args: Union[Message, Sequence[Message]]) -> List[Dict[str, Any]]:
        raise NotImplementedError("DeepSeek embeddings not supported.")

    async def alist_models(self) -> List[str]:
        raise NotImplementedError("DeepSeek embeddings not supported.")
