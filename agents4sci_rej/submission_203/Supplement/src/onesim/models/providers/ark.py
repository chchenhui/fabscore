import asyncio
from typing import Any, Dict, List, Sequence, Union, Optional

from loguru import logger
from volcenginesdkarkruntime import Ark

from ..core.model_base import ModelAdapterBase
from ..core.model_response import ModelResponse
from ..core.message import Message


class ArkChatAdapter(ModelAdapterBase):
    """
    Adapter for Volcengine Ark Chat API, with synchronous and asynchronous support.
    """

    def __init__(
        self,
        config_name: str,
        model_name: str,
        api_key: str = None,
        client_args: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        generate_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(config_name=config_name, model_name=model_name, **kwargs)
        self.stream = stream
        self.generate_args = generate_args or {}
        client_args = client_args or {}
        # Ark doesn't need base_url
        try:
            self.client = Ark(api_key=api_key, **client_args)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ark client: {e}")

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        call_args = {**self.generate_args, **kwargs}
        self._validate_messages(messages)

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **call_args
            )
            assistant_message = resp.choices[0].message
            return ModelResponse(
                text=assistant_message,
                raw=resp.__dict__,
                usage=None,
                model_info={
                    "config_name": self.config_name,
                    "model_name": self.model_name
                }
            )
        except Exception as e:
            logger.error(f"Error calling Ark API: {e}")
            raise

    async def acall(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.__call__(messages, stream=stream, **kwargs)
        )

    def _validate_messages(self, messages: List[Dict[str, Any]]) -> None:
        if not isinstance(messages, list):
            raise ValueError(f"Ark messages must be a list, got {type(messages)}")
        for msg in messages:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dict")
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message dict must have 'role' and 'content'")

    def format(self, *args: Union[Message, Sequence[Message]]) -> List[Dict[str, Any]]:
        return ModelAdapterBase.format_for_common_chat_models(*args)

    def list_models(self) -> List[str]:
        return [self.model_name]

    async def alist_models(self) -> List[str]:
        return [self.model_name]


class ArkEmbeddingAdapter(ModelAdapterBase):
    """
    Adapter for Volcengine Ark Embedding API, with synchronous and asynchronous support.
    """

    def __init__(
        self,
        config_name: str,
        model_name: str,
        api_key: str = None,
        client_args: Optional[Dict[str, Any]] = None,
        generate_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(config_name=config_name, model_name=model_name, **kwargs)
        self.generate_args = generate_args or {}
        client_args = client_args or {}

        try:
            self.client = Ark(api_key=api_key, **client_args)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ark client: {e}")

    def __call__(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> ModelResponse:
        call_args = {**self.generate_args, **kwargs}
        # ensure list
        inputs = [texts] if isinstance(texts, str) else list(texts)

        try:
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=inputs,
                encoding_format=call_args.get('encoding_format', 'float'),
                **{k: v for k, v in call_args.items() if k != 'encoding_format'}
            )
            embeddings = [item.embedding for item in resp.data]
            return ModelResponse(
                embedding=embeddings,
                raw=resp.__dict__,
                usage=None,
                model_info={
                    "config_name": self.config_name,
                    "model_name": self.model_name
                }
            )
        except Exception as e:
            logger.error(f"Error calling Ark Embedding API: {e}")
            raise

    async def acall(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> ModelResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.__call__(texts, **kwargs)
        )

    def format(self, *args: Union[str, Message, Sequence[Union[str, Message]]]) -> List[str]:
        texts: List[str] = []
        for arg in args:
            if isinstance(arg, str):
                texts.append(arg)
            elif isinstance(arg, Message):
                texts.append(str(arg.content))
            elif hasattr(arg, '__iter__'):
                texts.extend(self.format(*arg))
            else:
                raise TypeError(f"Expected str or Message, got {type(arg)}")
        return texts

    def list_models(self) -> List[str]:
        return [self.model_name]

    async def alist_models(self) -> List[str]:
        return [self.model_name]
