import asyncio
import requests
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger
from openai import OpenAI, AsyncOpenAI

from ..core.model_base import ModelAdapterBase
from ..core.model_response import ModelResponse
from ..core.message import Message


class AliyunChatAdapter(ModelAdapterBase):
    """
    Adapter for Alibaba Cloud DashScope Chat API (OpenAI‐compatible).
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
        Initialize the DashScope chat adapter.

        Args:
            config_name: Identifier for this adapter instance.
            model_name: DashScope model ID (e.g. "qwen-plus").
            api_key: Your DashScope API key.
            client_args: Extra OpenAI client params (timeout, retries…).
            stream: Whether to request streaming (DashScope may ignore).
            generate_args: Default generation params (temperature, max_tokens…).
            **kwargs: Additional params passed to base.
        """
        super().__init__(config_name=config_name, model_name=model_name, **kwargs)
        self.stream = stream
        self.generate_args = generate_args or {}
        # set DashScope base_url, but allow override via client_args["base_url"]
        client_args = client_args or {}
        default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        client_args.setdefault("base_url", default_base_url)
        # Initialize the OpenAI client
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Please install it using: pip install openai"
            )

        self.client = openai.OpenAI(api_key=api_key, **client_args)

        # Initialize the OpenAI async client
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.warning(
                "OpenAI async package not found. Async calls will use the sync client in an event loop."
            )
            self.async_client = None
        else:
            self.async_client = AsyncOpenAI(api_key=api_key, **client_args)

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Synchronous chat completion.

        Args:
            messages: List of {"role","content"} dicts.
            stream: Streaming flag, may be ignored.
            **kwargs: Override generate_args.

        Returns:
            ModelResponse with text, raw, usage, model_info.
        """
        call_kwargs = {**self.generate_args, **kwargs}
        self._validate_messages(messages)
        use_stream = self.stream if stream is None else stream
        call_kwargs.update({
            "model": self.model_name,
            "messages": messages,
            "stream": use_stream
        })

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
            logger.error(f"Aliyun chat sync call failed: {e}")
            raise

    async def acall(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Asynchronous chat completion via thread pool or AsyncOpenAI.

        Args and return as for __call__.
        """
        call_kwargs = {**self.generate_args, **kwargs}
        self._validate_messages(messages)
        use_stream = self.stream if stream is None else stream
        call_kwargs.update({
            "model": self.model_name,
            "messages": messages,
            "stream": use_stream
        })

        try:
            if self.async_client:
                resp = await self.async_client.chat.completions.create(**call_kwargs)
            else:
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None, lambda: self.client.chat.completions.create(**call_kwargs)
                )
            content = resp.choices[0].message.content
            return ModelResponse(
                text=content,
                raw=resp.model_dump(),
                usage=getattr(resp, "usage", None),
                model_info={"config_name": self.config_name, "model_name": self.model_name}
            )
        except Exception as e:
            print(self.client.api_key)
            logger.error(f"Aliyun chat async call failed: {e}")
            raise

    def list_models(self) -> List[str]:
        """
        Synchronously call DashScope /deployments/models to list available models.
        """
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        try:
            resp = requests.get(self._list_url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("output", {}).get("models", [])
            return [item["model_name"] for item in items]
        except Exception as e:
            logger.error(f"Failed to list Aliyun models: {e}")
            raise

    async def alist_models(self) -> List[str]:
        """
        Asynchronous version of list_models.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.list_models)

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> None:
        if not isinstance(messages, list):
            raise ValueError(f"Messages must be a list, got {type(messages)}")
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Each message must be a dict with 'role' and 'content'")

    def format(self, *args: Union[Message, Sequence[Message]]) -> List[Dict[str, Any]]:
        """
        Format Message objects to the common {"role","content"} format.
        """
        return ModelAdapterBase.format_for_common_chat_models(*args)


class AliyunEmbeddingAdapter(ModelAdapterBase):
    """
    Adapter for Alibaba Cloud DashScope Embedding API.
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
        """
        Initialize the DashScope embedding adapter.

        Args:
            config_name: Identifier for this adapter.
            model_name: Embedding model ID (e.g. "text-embedding-v4").
            api_key: Your DashScope API key.
            client_args: Extra OpenAI client params.
            generate_args: Embedding params (dimensions, encoding_format…).
            **kwargs: Additional parameters.
        """
        super().__init__(config_name=config_name, model_name=model_name, **kwargs)
        self.generate_args = generate_args or {}
        client_args = {**(client_args or {}), "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"}

        self._api_key = api_key
        self._list_url = "https://dashscope.aliyuncs.com/api/v1/deployments/models"

        # sync OpenAI‐compatible client
        self.client = OpenAI(api_key=api_key, **client_args)
        # async client if available
        try:
            self.async_client = AsyncOpenAI(api_key=api_key, **client_args)
        except ImportError:
            logger.warning("AsyncOpenAI not available; using thread pool for acall.")
            self.async_client = None

    def __call__(self, texts: Union[str, List[str]], **kwargs) -> ModelResponse:
        """
        Synchronous embedding generation.

        Args:
            texts: Single string or list of strings.
            **kwargs: Override generate_args.

        Returns:
            ModelResponse with .embedding list.
        """
        call_kwargs = {**self.generate_args, **kwargs}
        if isinstance(texts, str):
            texts = [texts]
        try:
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                **call_kwargs
            )
            data = resp.model_dump()
            embeddings = [item["embedding"] for item in data.get("data", [])]
            return ModelResponse(embedding=embeddings, raw=data)
        except Exception as e:
            logger.error(f"Aliyun embedding sync call failed: {e}")
            raise

    async def acall(self, texts: Union[str, List[str]], **kwargs) -> ModelResponse:
        """
        Asynchronous embedding generation via thread pool or async client.
        """
        call_kwargs = {**self.generate_args, **kwargs}
        if isinstance(texts, str):
            texts = [texts]
        try:
            if self.async_client:
                resp = await self.async_client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    **call_kwargs
                )
                data = resp.model_dump()
            else:
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: self.client.embeddings.create(
                        model=self.model_name,
                        input=texts,
                        **call_kwargs
                    )
                )
                data = resp.model_dump()
            embeddings = [item["embedding"] for item in data.get("data", [])]
            return ModelResponse(embedding=embeddings, raw=data)
        except Exception as e:
            logger.error(f"Aliyun embedding async call failed: {e}")
            raise

    def list_models(self) -> List[str]:
        """
        Reuse chat adapter's list_models to fetch available models.
        """
        # same endpoint for chat and embedding
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        try:
            resp = requests.get(self._list_url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("output", {}).get("models", [])
            return [item["model_name"] for item in items]
        except Exception as e:
            logger.error(f"Failed to list Aliyun models: {e}")
            raise

    async def alist_models(self) -> List[str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.list_models)

    def format(self, *args: Union[str, Message, Sequence[Union[str, Message]]]) -> List[str]:
        """
        Format inputs for embedding: extract text from Message or use strings.
        """
        texts: List[str] = []
        from ..core.message import Message as _Msg
        for arg in args:
            if isinstance(arg, str):
                texts.append(arg)
            elif isinstance(arg, _Msg):
                texts.append(str(arg.content))
            elif isinstance(arg, (list, tuple)):
                texts.extend(self.format(*arg))  # type: ignore
            else:
                raise TypeError(f"Expected str, Message, or sequence, got {type(arg)}")
        return texts
