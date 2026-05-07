"""
This module defines the Message class for representing chat messages.
"""

from typing import Any, Dict, List, Optional, Union


class Message:
    """
    Represents a chat message for model interaction.
    
    This class defines a structured format for messages sent to and from models,
    with support for different roles, content types, and metadata.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        content: Any = None,
        role: str = "user",
        images: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize a message.
        
        Args:
            content: The main content of the message.
            role: The role of the message sender (e.g., "system", "user", "assistant").
            name: Optional name of the sender.
            images: Optional list of image file paths to include with the message.
            **kwargs: Additional metadata to store with the message.
        """
        self.role = role
        self.content = content
        self.name = name
        self.images = images
        self.metadata = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.
        
        Returns:
            Dict representing the message.
        """
        result = {
            "role": self.role,
            "content": self.content
        }
        
        if self.name:
            result["name"] = self.name
            
        if self.images:
            result["images"] = self.images
            
        if self.metadata:
            result.update(self.metadata)
            
        return result
    
    def __str__(self) -> str:
        """String representation of the message."""
        prefix = f"{self.name or self.role}: "
        if isinstance(self.content, str):
            return prefix + self.content
        return prefix + str(self.content)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """
        Create a message from a dictionary.
        
        Args:
            data: Dictionary containing message data.
            
        Returns:
            Message: A new Message instance.
        """
        role = data.pop("role", "user")
        content = data.pop("content", "")
        name = data.pop("name", None)
        images = data.pop("images", None)
        
        return cls(role=role, content=content, name=name, images=images, **data)


# Define common message types for convenience
def SystemMessage(content: Any, **kwargs) -> Message:
    """Create a system message."""
    return Message(role="system", content=content, **kwargs)


def UserMessage(content: Any, **kwargs) -> Message:
    """Create a user message."""
    return Message(role="user", content=content, **kwargs)


def AssistantMessage(content: Any, **kwargs) -> Message:
    """Create an assistant message."""
    return Message(role="assistant", content=content, **kwargs) 