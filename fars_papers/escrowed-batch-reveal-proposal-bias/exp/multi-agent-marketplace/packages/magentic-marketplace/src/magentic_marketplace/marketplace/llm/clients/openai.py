"""OpenAI model client implementation."""

import copy
import json
import threading
from collections.abc import Sequence
from hashlib import sha256
from typing import Any, Literal, cast, overload


_GEMINI_UNSUPPORTED_KEYS = {"title", "default", "$defs"}


def _gemini_compatible_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a pydantic JSON schema to one Gemini function-calling accepts.

    Handles: $defs/$ref inlining, const->enum, anyOf->nullable, title/default removal.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node.pop("$ref")
                ref_name = ref_path.rsplit("/", 1)[-1]
                if ref_name in defs:
                    resolved = copy.deepcopy(defs[ref_name])
                    resolved.update({k: v for k, v in node.items() if k != "$ref"})
                    return _resolve(resolved)
                return node

            if "const" in node:
                val = node.pop("const")
                node["enum"] = [val]
                if "type" not in node:
                    node["type"] = type(val).__name__

            if "anyOf" in node:
                variants = node["anyOf"]
                non_null = [v for v in variants if v != {"type": "null"} and v.get("type") != "null"]
                if len(non_null) == 1:
                    merged = _resolve(non_null[0])
                    merged["nullable"] = True
                    for k, v in node.items():
                        if k not in ("anyOf",):
                            merged.setdefault(k, v)
                    return {k: v for k, v in merged.items() if k not in _GEMINI_UNSUPPORTED_KEYS}
                else:
                    node.pop("anyOf")
                    node["type"] = "string"

            result = {}
            for k, v in node.items():
                if k in _GEMINI_UNSUPPORTED_KEYS:
                    continue
                result[k] = _resolve(v)
            return result

        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)

import pydantic
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.shared_params import FunctionDefinition

from ..base import (
    AllowedChatCompletionMessageParams,
    ProviderClient,
    TResponseModel,
    Usage,
)
from ..config import BaseLLMConfig, EnvField


class OpenAIConfig(BaseLLMConfig):
    """Configuration for OpenAI provider."""

    provider: Literal["openai"] = EnvField("LLM_PROVIDER", default="openai")  # pyright: ignore[reportIncompatibleVariableOverride]
    api_key: str = EnvField("OPENAI_API_KEY", exclude=True)
    base_url: str | None = EnvField("OPENAI_BASE_URL", default=None)


class OpenAIClient(ProviderClient[OpenAIConfig]):
    """OpenAI model client that accepts OpenAI SDK arguments."""

    _client_cache: dict[str, "OpenAIClient"] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: OpenAIConfig | None = None):
        """Initialize OpenAI client.

        Args:
            config: OpenAI configuration. If None, creates from environment.

        """
        if config is None:
            config = OpenAIConfig()
        else:
            config = OpenAIConfig.model_validate(config)

        super().__init__(config)

        self.config = config
        if not self.config.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config."
            )
        self.client = AsyncOpenAI(
            api_key=self.config.api_key, base_url=self.config.base_url
        )

    @staticmethod
    def _get_cache_key(config: OpenAIConfig) -> str:
        """Generate cache key for a config."""
        config_json = config.model_dump_json(include={"api_key", "provider"})
        return sha256(config_json.encode()).hexdigest()

    @staticmethod
    def from_cache(config: OpenAIConfig) -> "OpenAIClient":
        """Get or create client from cache."""
        cache_key = OpenAIClient._get_cache_key(config)
        with OpenAIClient._cache_lock:
            if cache_key not in OpenAIClient._client_cache:
                OpenAIClient._client_cache[cache_key] = OpenAIClient(config)
            return OpenAIClient._client_cache[cache_key]

    @overload
    async def _generate(
        self,
        *,
        model: str,
        messages: Sequence[AllowedChatCompletionMessageParams],
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | int | None = None,
        response_format: None = None,
        **kwargs: Any,
    ) -> tuple[str, Usage]: ...

    @overload
    async def _generate(
        self,
        *,
        model: str,
        messages: Sequence[AllowedChatCompletionMessageParams],
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | int | None = None,
        response_format: type[TResponseModel],
        **kwargs: Any,
    ) -> tuple[TResponseModel, Usage]: ...

    async def _generate(
        self,
        *,
        model: str,
        messages: Sequence[AllowedChatCompletionMessageParams],
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | int | None = None,
        response_format: type[TResponseModel] | None = None,
        **kwargs: Any,
    ) -> tuple[str, Usage] | tuple[TResponseModel, Usage]:
        """Generate completion using OpenAI API."""
        # Build arguments, handling reasoning models
        args: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        # Handle reasoning vs non-reasoning models
        is_reasoning_model = any(
            reasoning_model in model for reasoning_model in ("gpt-5", "o4", "o3", "o1")
        )

        if is_reasoning_model:
            # Reasoning models use max_completion_tokens
            if max_tokens:
                args["max_completion_tokens"] = max_tokens

            if "gpt-5-chat" not in model:
                # Most reasoning models don't support temperature < 1
                if temperature and temperature < 1.0:
                    temperature = None

            # Handle reasoning effort for supported models
            if reasoning_effort is not None and "o1" not in model:
                if reasoning_effort == "minimal":
                    reasoning_effort = "low"  # o models don't support minimal
                args["reasoning_effort"] = reasoning_effort
        else:
            # Non-reasoning models
            if temperature is not None:
                args["temperature"] = temperature
            if max_tokens is not None:
                args["max_tokens"] = max_tokens

        # Add any additional kwargs
        args.update(kwargs)

        # Handle structured output
        if response_format is not None:
            _is_native_openai = any(
                p in model for p in ("gpt-", "o1", "o3", "o4", "chatgpt")
            )
            if not _is_native_openai:
                struct_kwargs: dict[str, Any] = {}
                if temperature is not None:
                    struct_kwargs["temperature"] = temperature
                if max_tokens is not None:
                    struct_kwargs["max_tokens"] = max_tokens
                return await self._generate_struct(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    **struct_kwargs,
                )

            # Make 3 attempts to parse the model
            exceptions: list[Exception] = []
            for _ in range(3):
                try:
                    response = await self.client.chat.completions.parse(
                        response_format=response_format, **args
                    )
                    parsed = response.choices[0].message.parsed
                    if parsed is not None:
                        usage = Usage(
                            token_count=response.usage.total_tokens
                            if response.usage
                            else 0,
                            provider="openai",
                            model=model,
                        )
                        return parsed, usage
                    elif response.choices[0].message.refusal:
                        # Present information to retry
                        raise ValueError(response.choices[0].message.refusal)
                    else:
                        # Unknown failure, no info to retry on, break
                        break

                except Exception as e:
                    exceptions.append(e)
                    # Append the error message to the chat history so that the model can retry with info
                    args["messages"].append({"role": "user", "content": str(e)})
            # if we make it here, we exhausted our retries
            exc_message = "Exceeded attempts to parse response_format."
            if exceptions:
                exc_message += "Inner exceptions: " + " ".join(map(str, exceptions))
            raise RuntimeError(exc_message)

        else:
            # Regular completion
            response = cast(
                ChatCompletion, await self.client.chat.completions.create(**args)
            )
            usage = Usage(
                token_count=response.usage.total_tokens if response.usage else 0,
                provider="openai",
                model=model,
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content, usage
            return "", usage

    async def _generate_text(
        self,
        model: str,
        messages: Sequence[AllowedChatCompletionMessageParams],
        **kwargs: Any,
    ):
        response = await self.client.chat.completions.create(
            model=model, messages=messages, stream=False, **kwargs
        )
        usage = Usage(
            token_count=response.usage.total_tokens if response.usage else 0,
            provider="openai",
            model=model,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content, usage
        return "", usage

    async def _generate_struct(
        self,
        model: str,
        messages: Sequence[ChatCompletionMessageParam],
        response_format: type[TResponseModel],
        **kwargs: Any,
    ):
        messages = list(messages)
        # Track exception messages for final error
        exceptions: list[str] = []

        raw_schema = response_format.model_json_schema()
        parameters = _gemini_compatible_schema(raw_schema)

        tool = ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=f"Generate{response_format.__name__}",
                description=f"Generate a {response_format.__name__}.",
                parameters=parameters,
            ),
        )

        # Make 3 attempts while recovering from errors.
        for _ in range(3):
            completion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[tool],
                tool_choice="required",
                stream=False,
                **kwargs,
            )

            if not completion.choices:
                raise ValueError("Failed to _generate_struct: choices was empty")

            message = completion.choices[0].message
            tool_calls = message.tool_calls

            if not tool_calls:
                raise ValueError("Failed to _generate_struct: tool_calls was empty")

            tool_call = tool_calls[0]

            if tool_call.type != "function":
                raise ValueError(
                    "Failed to _generate_struct: tool_call was not function type"
                )

            # Append generated tool call to message history for error recovery if parsing fails.
            messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                )
            )

            try:
                response: TResponseModel = response_format.model_validate_json(
                    tool_call.function.arguments
                )
                usage = Usage(
                    token_count=completion.usage.total_tokens if completion.usage else 0,
                    provider="openai",
                    model=model,
                )
                return response, usage
            except (json.decoder.JSONDecodeError, pydantic.ValidationError) as e:
                import traceback

                tb_str = traceback.format_exc()
                error_message = f"Error parsing tool: {e}\nTraceback:\n{tb_str}"
                exceptions.append(error_message)
                messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=tool_call.id,
                        content=error_message,
                    )
                )
        raise Exception(
            "Failed to _generate_struct: Exceeded maximum retries. Inner exceptions: "
            + " -> ".join(exceptions)
        )
