"""
This module defines the base model adapter class for different language models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

from .model_response import ModelResponse
from .message import Message
from loguru import logger


class ModelAdapterBase(ABC):
    """
    Base class for all model adapters.
    
    This abstract class defines the interface that all model adapters must implement,
    providing a consistent way to interact with different language models.
    
    Model implementations can be called either synchronously or asynchronously:
    - For synchronous calls: model(...) or model.__call__(...)
    - For asynchronous calls: await model.acall(...)
    """

    def __init__(self, config_name: str, model_name: str = None, **kwargs):
        """
        Initialize the model adapter.
        
        Args:
            config_name: The identifier for this model configuration.
            model_name: The specific model name (if different from config_name).
            **kwargs: Additional model-specific parameters.
        """
        self.config_name = config_name
        self.model_name = model_name or config_name
        self.max_length = kwargs.get("max_length", None)
        
        # Store initialization args for logging
        self._init_args = {
            "config_name": config_name,
            "model_name": self.model_name,
            **kwargs
        }
        
        # Token usage tracking
        self._token_tracker = None
        
    @property
    def token_tracker(self):
        """Get the token tracker instance, initializing it if necessary."""
        if self._token_tracker is None:
            try:
                from ..utils.token_usage import get_token_tracker
                self._token_tracker = get_token_tracker()
            except ImportError:
                logger.debug("Token tracking module not available")
                return None
        return self._token_tracker
        
    def _track_token_usage(self, usage_data: Dict[str, Any], model_name: Optional[str] = None):
        """
        Track token usage from response data.
        
        Args:
            usage_data: Dictionary containing token usage information
            model_name: Name of the model to associate with usage
        """
        if not usage_data or not self.token_tracker:
            return
            
        model_name = model_name or self.model_name
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)
        
        try:
            self.token_tracker.track(
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        except Exception as e:
            logger.error(f"Error tracking token usage: {e}")
            
    def _add_model_info_to_response(self, response: ModelResponse) -> ModelResponse:
        """
        Add model information to a response object.
        
        Args:
            response: The model response to augment
            
        Returns:
            The updated model response
        """
        if not hasattr(response, 'model_info'):
            return response
            
        # If model_info is not set, initialize it
        if not response.model_info:
            response.model_info = {}
            
        # Add model information
        response.model_info.update({
            "model_name": self.model_name,
            "config_name": self.config_name
        })
        
        return response
        
    def __call__(self, *args, **kwargs) -> ModelResponse:
        """
        Process inputs and generate a model response synchronously.
        
        This synchronous method should be implemented by all subclasses to handle
        model-specific input processing and inference.
        
        Args:
            *args: Input arguments, typically includes formatted messages.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            ModelResponse: The standardized response from the model.
        """
        raise NotImplementedError(
            f"Model adapter {type(self).__name__} must implement __call__ method"
        )
    
    async def acall(self, *args, **kwargs) -> ModelResponse:
        """
        Process inputs and generate a model response asynchronously.
        
        This async method should be implemented by all subclasses to handle
        model-specific input processing and inference.
        
        Args:
            *args: Input arguments, typically includes formatted messages.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            ModelResponse: The standardized response from the model.
        """
        raise NotImplementedError(
            f"Model adapter {type(self).__name__} does not implement asynchronous acall method"
        )
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """
        Synchronously return the list of available model IDs
        from this adapter’s backend.

        Returns:
            A list of model identifier strings.
        """
        pass

    @abstractmethod
    async def alist_models(self) -> List[str]:
        """
        Asynchronously return the list of available model IDs
        from this adapter’s backend.

        Returns:
            A list of model identifier strings.
        """
        pass
        
    @abstractmethod
    def format(self, *args: Union[Message, Sequence[Message]]) -> Any:
        """
        Format input messages into the structure expected by the model.
        
        Args:
            *args: Input messages or sequences of messages.
            
        Returns:
            The formatted messages ready for model input.
        """
        pass
    
    @staticmethod
    def format_for_common_chat_models(*args: Union[Message, Sequence[Message]]) -> List[Dict[str, str]]:
        """
        Format messages for standard chat models that follow the common format.
        
        Args:
            *args: Input messages or sequences of messages.
            
        Returns:
            List[Dict[str, str]]: A list of message dictionaries in the format:
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        """
        messages = []
        for arg in args:
            if arg is None:
                continue
                
            if isinstance(arg, Message):
                msg_dict = {
                    "role": arg.role,
                    "content": ModelAdapterBase._convert_to_str(arg.content)
                }
                # Only include name if it exists
                if arg.name:
                    msg_dict["name"] = arg.name
                
                messages.append(msg_dict)
            elif isinstance(arg, (list, tuple)):
                # Recursively format sequences
                messages.extend(ModelAdapterBase.format_for_common_chat_models(*arg))
            else:
                raise TypeError(
                    f"Expected Message or sequence of Messages, got {type(arg)}"
                )
        
        return messages
    
    @staticmethod
    def _convert_to_str(content: Any) -> str:
        """Convert content to string format."""
        if content is None:
            return ""
        if isinstance(content, (str, int, float, bool)):
            return str(content)
        if hasattr(content, "__str__"):
            return str(content)
        return repr(content)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this model instance.
        
        Returns:
            Dict containing model configuration details.
        """
        return {
            "config_name": self.config_name,
            "model_name": self.model_name,
            "model_type": self.__class__.__name__,
            "max_length": self.max_length
        } 