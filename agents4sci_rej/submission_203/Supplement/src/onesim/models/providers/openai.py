"""
Implementation of OpenAI chat model adapter with both sync and async support.
"""

import base64
import json
import os
from typing import Any, Dict, Generator, List, Optional, Sequence, Union
import asyncio

from loguru import logger

from ..core.model_base import ModelAdapterBase
from ..core.model_response import ModelResponse
from ..core.message import Message


class OpenAIChatAdapter(ModelAdapterBase):
    """
    Adapter for OpenAI chat models and compatible APIs.
    
    This class provides an interface to OpenAI's chat completions API
    and other APIs that follow the same format (like local deployments
    using vLLM, FastChat, etc). Supports both synchronous and asynchronous calls.
    """

    def __init__(
        self,
        config_name: str,
        model_name: str = None,
        api_key: str = None,
        organization: str = None,
        client_args: dict = None,
        stream: bool = False,
        generate_args: dict = None,
        **kwargs
    ):
        """
        Initialize the OpenAI chat model adapter.
        
        Args:
            config_name: Configuration name for this model.
            model_name: The name of the model (e.g., "gpt-4", "llama-2-70b").
            api_key: OpenAI API key or compatible API key.
            organization: Organization ID (for OpenAI organization access).
            client_args: Additional arguments for the OpenAI client.
            stream: Whether to stream responses by default.
            generate_args: Default parameters for generation requests.
            **kwargs: Additional parameters.
        """
        super().__init__(config_name=config_name, model_name=model_name, **kwargs)

        self.stream = stream
        self.generate_args = generate_args or {}

        # Initialize the OpenAI client
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Please install it using: pip install openai"
            )
        # FEAT:fixed API address
        client_args = client_args or {}
        default_base_url = "https://api.openai.com/v1/"
        client_args.setdefault("base_url", default_base_url)

        self.client = openai.OpenAI(
            api_key=api_key,
            organization=organization,
            **client_args
        )

        # Initialize the OpenAI async client
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.warning(
                "OpenAI async package not found. Async calls will use the sync client in an event loop."
            )
            self.async_client = None
        else:
            self.async_client = AsyncOpenAI(
                api_key=api_key,
                organization=organization,
                **client_args
            )

    def __call__(
        self,
        messages: List[Dict],
        stream: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Synchronous call to the OpenAI chat completions API.
        
        Args:
            messages: List of formatted message dictionaries.
            stream: Whether to stream the response.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            ModelResponse: The response from the model.
        """
        # Merge default arguments with provided ones
        call_kwargs = {**self.generate_args, **kwargs}

        # Validate messages format
        self._validate_messages(messages)

        # Determine streaming mode
        use_stream = self.stream if stream is None else stream

        # Set up API call parameters
        call_kwargs.update({
            "model": self.model_name,
            "messages": messages,
            "stream": use_stream
        })

        if use_stream:
            call_kwargs["stream_options"] = {"include_usage": True}

        try:
            # Make the API call
            response = self.client.chat.completions.create(**call_kwargs)

            if use_stream:
                # Handle streaming response
                def generate_stream():
                    text = ""
                    last_chunk = {}
                    for chunk in response:
                        chunk_data = chunk.model_dump()
                        if self._has_content_in_delta(chunk_data):
                            content = chunk_data["choices"][0]["delta"]["content"]
                            text += content
                            yield text
                        last_chunk = chunk_data

                    # Track token usage from the last chunk if available
                    if "usage" in last_chunk:
                        self._track_token_usage(last_chunk["usage"])

                return ModelResponse(
                    stream=generate_stream(),
                    model_info={
                        "model_name": self.model_name,
                        "config_name": self.config_name
                    }
                )
            else:
                # Handle normal response
                response_data = response.model_dump()

                # Track token usage
                if "usage" in response_data:
                    self._track_token_usage(response_data["usage"])

                if self._has_content_in_message(response_data):
                    content = response_data["choices"][0]["message"]["content"]
                    return ModelResponse(
                        text=content, 
                        raw=response_data,
                        usage=response_data.get("usage"),
                        model_info={
                            "model_name": self.model_name,
                            "config_name": self.config_name
                        }
                    )
                else:
                    raise ValueError(f"Invalid response format: {response_data}")

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

    async def acall(
        self,
        messages: List[Dict],
        stream: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Asynchronous call to the OpenAI chat completions API.
        
        Args:
            messages: List of formatted message dictionaries.
            stream: Whether to stream the response.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            ModelResponse: The response from the model.
        """
        # Merge default arguments with provided ones
        call_kwargs = {**self.generate_args, **kwargs}

        # Validate messages format
        self._validate_messages(messages)

        # Determine streaming mode
        use_stream = self.stream if stream is None else stream

        # Set up API call parameters
        call_kwargs.update({
            "model": self.model_name,
            "messages": messages,
            "stream": use_stream
        })

        if use_stream:
            call_kwargs["stream_options"] = {"include_usage": True}

        try:
            # Use async client if available, otherwise run sync client in thread
            if self.async_client:
                response = await self.async_client.chat.completions.create(**call_kwargs)
            else:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.client.chat.completions.create(**call_kwargs)
                )

            if use_stream:
                # Handle streaming response
                async def generate_stream_async():
                    text = ""
                    last_chunk = {}

                    async for chunk in response:
                        chunk_data = chunk.model_dump()

                        # Check if the chunk contains content
                        if self._has_content_in_delta(chunk_data):
                            content = chunk_data["choices"][0]["delta"]["content"]
                            text += content
                            yield text

                        last_chunk = chunk_data

                    if "usage" in last_chunk:
                        self._track_token_usage(last_chunk["usage"])
                    # Ensure the last chunk has the complete message
                    if "choices" not in last_chunk or not last_chunk["choices"]:
                        last_chunk["choices"] = [{}]

                    # Set the full text in the last chunk for reference
                    last_chunk["choices"][0]["message"] = {
                        "role": "assistant",
                        "content": text
                    }

                return ModelResponse(astream=generate_stream_async())
            else:
                # Handle non-streaming response
                response_data = response.model_dump()

                # Track token usage
                if "usage" in response_data:
                    # Record usage information in the tracker
                    self._track_token_usage(response_data["usage"])

                if self._has_content_in_message(response_data):
                    content = response_data["choices"][0]["message"]["content"]

                    # Create response with usage and model info
                    model_response = ModelResponse(
                        text=content, 
                        raw=response_data,
                        usage=response_data.get("usage"),
                        model_info={
                            "model_name": self.model_name,
                            "config_name": self.config_name
                        }
                    )

                    return model_response
                else:
                    raise ValueError(f"Invalid response format: {response_data}")

        except Exception as e:
            logger.error(f"Error calling OpenAI API asynchronously: {str(e)}")
            raise

    def list_models(self) -> List[str]:
        """
        Synchronously retrieve the list of available model IDs
        from the OpenAI-compatible endpoint.
        
        Returns:
            A list of model identifier strings, e.g. ["gpt-4", "gpt-3.5-turbo", …].
        """
        try:
            response = self.client.models.list()
            data = response.model_dump()
            # Extract the 'id' field from each model entry
            return [entry["id"] for entry in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise

    async def alist_models(self) -> List[str]:
        """
        Asynchronously retrieve the list of available model IDs
        from the OpenAI-compatible endpoint.
        
        Returns:
            A list of model identifier strings.
        """
        try:
            if self.async_client:
                response = await self.async_client.models.list()
            else:
                # Fallback to running the sync call in a thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.list()
                )
            data = response.model_dump()
            return [entry["id"] for entry in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to asynchronously list models: {e}")
            raise

    def _validate_messages(self, messages: List[Dict]) -> None:
        """Validate message format."""
        if not isinstance(messages, list):
            raise ValueError(
                f"OpenAI messages must be a list, got {type(messages)}"
            )

        if not all("role" in msg for msg in messages):
            raise ValueError(
                "Each message must contain a 'role' key"
            )

        # For content, we need to check if it's either a string or a list of content parts
        for msg in messages:
            if "content" not in msg and not isinstance(msg.get("content"), (str, list, type(None))):
                raise ValueError(
                    "Message content must be a string, list of content parts, or None"
                )

    def format(self, *args: Union[Message, Sequence[Message]]) -> List[Dict]:
        """
        Format messages for the OpenAI chat completions API.
        
        Args:
            *args: Messages or sequences of messages.
            
        Returns:
            List of formatted message dictionaries.
        """
        messages = []
        for arg in args:
            if arg is None:
                continue

            if isinstance(arg, Message):
                # Process message based on content type and whether images are present
                if arg.images:
                    # Handle messages with images
                    if isinstance(arg.content, list):
                        # Content is already a list, add images to it
                        content_parts = arg.content.copy()
                    else:
                        # Convert content to text first
                        content_parts = [{
                            "type": "text",
                            "text": self._convert_to_str(arg.content)
                        }]

                    # Add each image as an image_url part
                    for image_path in arg.images:
                        if os.path.exists(image_path):
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{self._get_image_extension(image_path)};base64,{self._encode_image(image_path)}",
                                }
                            })
                        else:
                            logger.warning(f"Image file not found: {image_path}")

                    msg_dict = {
                        "role": arg.role,
                        "content": content_parts
                    }
                elif isinstance(arg.content, list):
                    # Content is a list of content parts (text and images)
                    content_parts = []
                    for part in arg.content:
                        if isinstance(part, dict) and part.get("type") == "image_path":
                            # Convert image path to base64 and add as image_url content
                            image_path = part.get("path")
                            if image_path and os.path.exists(image_path):
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{self._get_image_extension(image_path)};base64,{self._encode_image(image_path)}",
                                    }
                                })
                            else:
                                logger.warning(f"Image file not found: {image_path}")
                        elif isinstance(part, dict) and part.get("type") in ["text", "image_url"]:
                            # Pass through other content part types
                            content_parts.append(part)
                        else:
                            # Convert plain content to text type
                            content_parts.append({
                                "type": "text",
                                "text": self._convert_to_str(part)
                            })

                    msg_dict = {
                        "role": arg.role,
                        "content": content_parts
                    }
                else:
                    # Standard message format with string content
                    msg_dict = {
                        "role": arg.role,
                        "content": self._convert_to_str(arg.content)
                    }

                # Only include name if it exists
                if arg.name:
                    msg_dict["name"] = arg.name

                messages.append(msg_dict)
            elif isinstance(arg, (list, tuple)):
                # Process sequences of messages
                messages.extend(self.format(*arg))
            else:
                raise TypeError(
                    f"Expected Message or sequence of Messages, got {type(arg)}"
                )

        return messages

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def _get_image_extension(image_path: str) -> str:
        """
        Get the image file extension.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            File extension without the dot (e.g., 'jpeg', 'png').
        """
        _, ext = os.path.splitext(image_path)
        if ext.lower() in ['.jpg', '.jpeg']:
            return 'jpeg'
        elif ext.lower() == '.png':
            return 'png'
        elif ext.lower() == '.webp':
            return 'webp'
        elif ext.lower() == '.gif':
            return 'gif'
        else:
            # Default to jpeg for unknown types
            return 'jpeg'

    @staticmethod
    def _has_content_in_delta(response: Dict) -> bool:
        """Check if a streaming response chunk contains content."""
        try:
            return (
                response.get("choices")
                and response["choices"][0].get("delta")
                and "content" in response["choices"][0]["delta"]
            )
        except (KeyError, IndexError):
            return False

    @staticmethod
    def _has_content_in_message(response: Dict) -> bool:
        """Check if a response contains a message with content."""
        try:
            return (
                response.get("choices")
                and response["choices"][0].get("message")
                and "content" in response["choices"][0]["message"]
            )
        except (KeyError, IndexError):
            return False

    @staticmethod
    def _convert_to_str(content: Any) -> str:
        """Convert content to string."""
        if isinstance(content, str):
            return content
        elif hasattr(content, "__str__"):
            return str(content)
        else:
            return json.dumps(content)

    @staticmethod
    def _convert_to_content(content: Any) -> Union[str, List[Dict]]:
        """
        Convert content to OpenAI format, handling both text and images.
        
        Args:
            content: Content to convert (text, image, or multimodal content)
            
        Returns:
            Formatted content for OpenAI API
        """
        # If content is already a list of content parts, return as is
        if isinstance(content, list) and all(isinstance(item, dict) for item in content):
            return content

        # Handle string content
        if isinstance(content, str):
            return content

        # Handle dict with image_url or other special content
        if isinstance(content, dict):
            # If content is a dict with image_url, format it for OpenAI
            if "image_url" in content:
                return [
                    {
                        "type": "image_url",
                        "image_url": content["image_url"]
                    }
                ]
            # If content has text and image_url, format as multimodal
            elif "text" in content and "image_url" in content:
                return [
                    {
                        "type": "text",
                        "text": content["text"]
                    },
                    {
                        "type": "image_url",
                        "image_url": content["image_url"]
                    }
                ]

        # For other types, use json.dumps as fallback
        return json.dumps(content)


class OpenAIEmbeddingAdapter(ModelAdapterBase):
    """
    Adapter for OpenAI embedding models and compatible APIs.
    
    This class provides an interface to OpenAI's embedding API
    and other APIs that follow the same format (like local deployments).
    Supports both synchronous and asynchronous calls.
    """

    def __init__(
        self,
        config_name: str,
        model_name: str = None,
        api_key: str = None,
        organization: str = None,
        client_args: dict = None,
        generate_args: dict = None,
        **kwargs
    ):
        """
        Initialize the OpenAI embedding model adapter.
        
        Args:
            config_name: Configuration name for this model.
            model_name: The name of the model (e.g., "text-embedding-ada-002").
            api_key: OpenAI API key or compatible API key.
            organization: Organization ID (for OpenAI organization access).
            client_args: Additional arguments for the OpenAI client.
            generate_args: Default parameters for generation requests.
            **kwargs: Additional parameters.
        """
        super().__init__(config_name=config_name, model_name=model_name, **kwargs)

        self.generate_args = generate_args or {}

        # Initialize the OpenAI client
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Please install it using: pip install openai"
            )

        client_args = client_args or {}
        self.client = openai.OpenAI(
            api_key=api_key,
            organization=organization,
            **client_args
        )

        # Initialize the OpenAI async client
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.warning(
                "OpenAI async package not found. Async calls will use the sync client in an event loop."
            )
            self.async_client = None
        else:
            self.async_client = AsyncOpenAI(
                api_key=api_key,
                organization=organization,
                **(client_args or {})
            )

    def __call__(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> ModelResponse:
        """
        Synchronous call to the OpenAI embedding API.
        
        Args:
            texts: Text or list of texts to embed.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            ModelResponse: The response containing embeddings.
        """
        # Merge default arguments with provided ones
        call_kwargs = {**self.generate_args, **kwargs}

        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Set up API call parameters
        call_kwargs.update({
            "model": self.model_name,
            "input": texts
        })

        try:
            # Make the API call
            response = self.client.embeddings.create(**call_kwargs)
            response_data = response.model_dump()

            # Extract embeddings
            if "data" in response_data and len(response_data["data"]) > 0:
                embeddings = [item["embedding"] for item in response_data["data"]]

                return ModelResponse(embedding=embeddings, raw=response_data)
            else:
                raise ValueError(f"Invalid response format: {response_data}")

        except Exception as e:
            logger.error(f"Error calling OpenAI Embedding API: {str(e)}")
            raise

    async def acall(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> ModelResponse:
        """
        Asynchronous call to the OpenAI embedding API.
        
        Args:
            texts: Text or list of texts to embed.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            ModelResponse: The response containing embeddings.
        """
        # Merge default arguments with provided ones
        call_kwargs = {**self.generate_args, **kwargs}

        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Set up API call parameters
        call_kwargs.update({
            "model": self.model_name,
            "input": texts
        })

        try:
            # Use async client if available, otherwise run sync client in thread
            if self.async_client:
                response = await self.async_client.embeddings.create(**call_kwargs)
            else:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.client.embeddings.create(**call_kwargs)
                )

            response_data = response.model_dump()

            # Extract embeddings
            if "data" in response_data and len(response_data["data"]) > 0:
                embeddings = [item["embedding"] for item in response_data["data"]]

                return ModelResponse(embedding=embeddings, raw=response_data)
            else:
                raise ValueError(f"Invalid response format: {response_data}")

        except Exception as e:
            logger.error(f"Error calling OpenAI Embedding API asynchronously: {str(e)}")
            raise

    def list_models(self) -> List[str]:
        """
        Synchronously retrieve the list of available model IDs
        from the OpenAI-compatible endpoint.
        
        Returns:
            A list of model identifier strings, e.g. ["gpt-4", "gpt-3.5-turbo", …].
        """
        try:
            response = self.client.models.list()
            data = response.model_dump()
            # Extract the 'id' field from each model entry
            return [entry["id"] for entry in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise

    async def alist_models(self) -> List[str]:
        """
        Asynchronously retrieve the list of available model IDs
        from the OpenAI-compatible endpoint.
        
        Returns:
            A list of model identifier strings.
        """
        try:
            if self.async_client:
                response = await self.async_client.models.list()
            else:
                # Fallback to running the sync call in a thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.list()
                )
            data = response.model_dump()
            return [entry["id"] for entry in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to asynchronously list models: {e}")
            raise

    def format(self, *args: Union[str, Message, Sequence[Union[str, Message]]]) -> List[str]:
        """
        Format inputs for the OpenAI embedding API.
        
        This method extracts text content from various input types.
        
        Args:
            *args: Strings, Messages, or sequences of these.
            
        Returns:
            List of text strings ready for embedding.
        """
        texts = []
        for arg in args:
            if arg is None:
                continue

            if isinstance(arg, str):
                # Plain text
                texts.append(arg)
            elif isinstance(arg, Message):
                # Extract text from Message
                texts.append(self._convert_to_str(arg.content))
            elif isinstance(arg, (list, tuple)):
                # Process sequences
                texts.extend(self.format(*arg))
            else:
                raise TypeError(
                    f"Expected str, Message, or sequence, got {type(arg)}"
                )

        return texts

    @staticmethod
    def _convert_to_str(content: Any) -> str:
        """Convert content to string."""
        if isinstance(content, str):
            return content
        elif hasattr(content, "__str__"):
            return str(content)
        else:
            return json.dumps(content)
