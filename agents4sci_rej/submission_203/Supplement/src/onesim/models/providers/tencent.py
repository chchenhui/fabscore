import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger
from openai import OpenAI, AsyncOpenAI

from ..core.model_base import ModelAdapterBase
from ..core.model_response import ModelResponse
from ..core.message import Message


class TencentChatAdapter(ModelAdapterBase):
    """
    Adapter for Tencent Hunyuan Chat API via OpenAI-compatible endpoint.
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
        Initialize the Tencent Hunyuan chat adapter.

        Args:
            config_name: Adapter config identifier.
            model_name: Hunyuan model ID, e.g. "hunyuan-turbos-latest".
            api_key: Your Hunyuan API key.
            client_args: Extra OpenAI client parameters.
            stream: Whether to request streaming.
            generate_args: Default generation parameters (e.g., extra_body).
            **kwargs: Other parameters.
        """
        super().__init__(config_name=config_name, model_name=model_name, **kwargs)
        self.stream = stream
        self.generate_args = generate_args or {}
        # ensure base_url points to Tencent Hunyuan endpoint
        client_args = client_args or {}
        default_base_url = "https://api.hunyuan.cloud.tencent.com/v1"
        client_args.setdefault("base_url", default_base_url)

        # sync client
        self.client = OpenAI(api_key=api_key, **client_args)
        # async client if available
        try:
            self.async_client = AsyncOpenAI(api_key=api_key, **client_args)
        except ImportError:
            logger.warning("AsyncOpenAI not available; acall will fallback to thread.")
            self.async_client = None

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Synchronous chat completion.

        Args:
            messages: [{"role","content"}] list.
            stream: Overrides self.stream.
            **kwargs: Override generate_args (e.g. extra_body).
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
            logger.error(f"Tencent chat sync call failed: {e}")
            raise

    async def acall(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Asynchronous chat completion, uses async client or thread pool fallback.
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
            logger.error(f"Tencent chat async call failed: {e}")
            raise

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Ensure messages is list of dict(role,content)."""
        if not isinstance(messages, list):
            raise ValueError(f"Messages must be list, got {type(messages)}")
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content'")

    def format(self, *args: Union[Message, Sequence[Message]]) -> List[Dict[str, Any]]:
        """Format Message objects into {"role","content"} dicts."""
        return ModelAdapterBase.format_for_common_chat_models(*args)

    def list_models(self) -> List[str]:
        """
        Tencent Hunyuan does not expose a public model-list endpoint.
        Return the configured model_name only.
        """
        return [self.model_name]

    async def alist_models(self) -> List[str]:
        """Async version of list_models."""
        return [self.model_name]


class TencentEmbeddingAdapter(ModelAdapterBase):
    """
    Adapter for Tencent Hunyuan Embedding API.

    Note: There is no dedicated Hunyuan embeddings endpoint in OpenAI-compatible SDK,
    use REST or SDK client.embeddings.create if supported.
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
        Initialize the Tencent Hunyuan embedding adapter.

        Args:
            config_name: Adapter config name.
            model_name: Embedding model ID, e.g. "hunyuan-embedding".
            api_key: Your Hunyuan API key.
            client_args: Extra client params.
            generate_args: Embedding parameters.
        """
        super().__init__(config_name=config_name, model_name=model_name, **kwargs)
        self.generate_args = generate_args or {}
        client_args = {
            **(client_args or {}),
            "base_url": "https://api.hunyuan.cloud.tencent.com/v1"
        }

        # sync client
        self.client = OpenAI(api_key=api_key, **client_args)
        # async client if available
        try:
            self.async_client = AsyncOpenAI(api_key=api_key, **client_args)
        except ImportError:
            logger.warning("AsyncOpenAI not available; using thread pool for acall.")
            self.async_client = None

    def __call__(self, texts: Union[str, List[str]], **kwargs) -> ModelResponse:
        """
        Synchronous embeddings.create call.

        Args:
            texts: Single string or list of strings.
            **kwargs: Override generate_args.
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
            logger.error(f"Tencent embedding sync call failed: {e}")
            raise

    async def acall(self, texts: Union[str, List[str]], **kwargs) -> ModelResponse:
        """
        Async embeddings call via async client or thread.
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
            logger.error(f"Tencent embedding async call failed: {e}")
            raise

    def format(self, *args: Union[str, Message, Sequence[Union[str, Message]]]) -> List[str]:
        """
        Format input texts or Message objects into list of strings.
        """
        texts: List[str] = []
        for arg in args:
            if isinstance(arg, str):
                texts.append(arg)
            elif isinstance(arg, Message):
                texts.append(str(arg.content))
            elif isinstance(arg, (list, tuple)):
                texts.extend(self.format(*arg))  # type: ignore
            else:
                raise TypeError(f"Expected str or Message, got {type(arg)}")
        return texts

    def list_models(self) -> List[str]:
        """
        No model-list endpoint for embeddings; return only configured model_name.
        """
        return [self.model_name]

    async def alist_models(self) -> List[str]:
        """Async version of list_models."""
        return [self.model_name]
