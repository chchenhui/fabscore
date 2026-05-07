import os
import os.path as osp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, Dict, cast
import openai 
import anthropic
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

from fabscore.utils.utils import encode_image

OPENAI_MODELS = [
    "o4-mini-2025-04-16",   
    "gpt-4.1-2025-04-14",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-5.1-2025-11-13",
    "gpt-5.2-2025-12-11",
    "gpt-5.5-2026-04-23",
]

ANTHROPIC_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-5-20251101",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
]

OPENROUTER_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b:free",
    "openai/gpt-oss-20b",
    "qwen/qwen3-235b-a22b-2507",
    "google/gemini-2.5-pro",
    "google/gemini-3-pro-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-pro-preview",
    "qwen/qwen3.5-plus-02-15",
    "qwen/qwen3.5-397b-a17b",
    "openai/gpt-5.5",
]

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
]

MAX_NUM_TOKENS = 4096


class LLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        pass


class OpenAILLM(LLM):
    """
    An LLM class for the OpenAI LLMs.
    """

    def __init__(
        self,
        model_name: str = "o4-mini-2025-04-16",
        api_key: Optional[str] = None,
        max_tokens: int = MAX_NUM_TOKENS,
        **kwargs: Any,
    ):
        if not api_key:
            self.client = openai.OpenAI()
        else:
            self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        # Parameter compatibility: GPT-5 uses max_completion_tokens; o1/o3/o4 do not use max_tokens
        is_reasoning_family = (
            model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4")
        )
        is_gpt5_family = model_name.startswith("gpt-5")

        if is_gpt5_family:
            if "max_completion_tokens" not in kwargs:
                kwargs["max_completion_tokens"] = max_tokens
        elif not is_reasoning_family:
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = max_tokens
        self.kwargs = kwargs

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text, optionally with a local image.

        image_path must be a local file path (e.g. /path/to/image.png).
        """
        tmp_kwargs = self.kwargs | kwargs
        if image_path:
            # Always treat image_path as a local file and embed it as
            # a data URL for OpenAI.
            image_url = (
                "data:image/jpeg;base64,"
                f"{encode_image(image_path)}"
            )
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]
        else:
            content = prompt
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": content}
                ],
            **tmp_kwargs
            )
        return response.choices[0].message.content


class AnthropicLLM(LLM):
    """
    An LLM class for the Anthropic LLMs.
    """

    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        max_tokens: int = MAX_NUM_TOKENS,
        **kwargs: Any,
    ):
        if not api_key:
            self.client = anthropic.Anthropic()
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        kwargs["max_tokens"] = max_tokens
        self.kwargs = kwargs

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text, optionally with a local image.

        image_path must be a local file path.
        """
        tmp_kwargs = self.kwargs | kwargs

        content = []
        if image_path:
            # Get media type from file extension (best-effort for local files)
            ext = osp.splitext(image_path)[1].lower()
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(ext, "image/jpeg")  # Default to jpeg

            source = {
                "type": "base64",
                "media_type": media_type,
                "data": encode_image(image_path),
            }

            content.append(
                {
                    "type": "image",
                    "source": source,
                }
            )

        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        messages = [{"role": "user", "content": content}]
        
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            **tmp_kwargs
        )
        return response.content[0].text


class OpenRouterLLM(OpenAILLM):
    """
    An LLM class for the OpenRouter LLMs.
    """

    def __init__(
        self,
        model_name: str = "google/gemini-2.5-pro",
        api_key: Optional[str] = None,
        max_tokens: int = MAX_NUM_TOKENS,
        **kwargs: Any,
    ):
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        base_url = "https://openrouter.ai/api/v1"
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_tokens
        self.kwargs = kwargs


class GeminiLLM(LLM):
    r"""An LLM class for the Gemini LLMs."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")

        # Create the client using the Google Genai client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        if "max_tokens" in kwargs and "max_output_tokens" not in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        self.kwargs = kwargs

    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text, optionally with an image.

        NOTE: For Gemini we only support **local image files** here.
        `image_path` must be a path on the local filesystem (e.g., \"cat.jpg\").
        """
        tmp_kwargs = self.kwargs | kwargs
        generation_config = self._create_generation_config(tmp_kwargs)

        # Build Gemini Content/Part structures so that images are properly
        # recognized as image parts, not just base64 strings.
        parts: list[types.Part] = [
            types.Part.from_text(text=prompt),
        ]

        if image_path:
            # Infer a concrete MIME type from the file extension.
            ext = osp.splitext(image_path)[1].lower()
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(ext, "image/jpeg")

            with open(image_path, "rb") as f:
                data = f.read()

            img_part = types.Part.from_bytes(
                data=data,
                mime_type=media_type,
            )
            parts.append(img_part)

        user_content = types.UserContent(parts=parts)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[user_content],
            config=generation_config,
        )
        # In the google-genai SDK, text responses are available via `.text`.
        return cast(str, response.text)

    
    def _create_generation_config(
        self, kwargs: Dict[str, Any]
    ) -> types.GenerateContentConfig:
        # Extract generation-specific parameters
        config_params = {}

        # Handle known parameters
        for param in [
            "max_output_tokens",
            "temperature",
            "top_p",
            "top_k",
            "response_mime_type",
            "stop_sequences",
            "candidate_count",
            "seed",
            "safety_settings",
            "system_instruction",
        ]:
            if param in kwargs:
                config_params[param] = kwargs[param]

        # Create a GenerateContentConfig object
        return types.GenerateContentConfig(**config_params)


def create_client(model_name, max_tokens=MAX_NUM_TOKENS):
    if model_name in OPENAI_MODELS:
        api_key = os.getenv("OPENAI_API_KEY")
        if model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
            reasoning_effort = "high"
            return OpenAILLM(model_name=model_name, api_key=api_key, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
        else:
            return OpenAILLM(model_name=model_name, api_key=api_key, max_tokens=max_tokens)
    elif model_name in ANTHROPIC_MODELS:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return AnthropicLLM(model_name=model_name, api_key=api_key, max_tokens=max_tokens)
    elif model_name in OPENROUTER_MODELS:
        api_key = os.getenv("OPENROUTER_API_KEY")
        return OpenRouterLLM(model_name=model_name, api_key=api_key, max_tokens=max_tokens)
    elif model_name in GEMINI_MODELS:
        api_key = os.getenv("GEMINI_API_KEY")
        return GeminiLLM(model_name=model_name, api_key=api_key, max_tokens=max_tokens)
    else:
        raise ValueError(f"Model {model_name} not supported.")


if __name__ == "__main__":
    llm = create_client(model_name="gpt-5.5-2026-04-23")
    response = llm.generate(
        prompt="What is the capital of France?",
    )
    print(response)