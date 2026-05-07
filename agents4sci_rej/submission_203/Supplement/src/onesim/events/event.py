from typing import List, Any, Optional, Dict, Union
from dataclasses import dataclass
import uuid
from enum import Enum
from onesim.utils.common import gen_id
import time
import datetime

class EventType(Enum):
    QUERY = "query"
    UPDATE = "update"
    
class Event:
    """Base event class for the agent communication system.
    
    All events in the system should inherit from this class. It provides standard
    fields like event_id, timestamp, source and target agent ids, and contains
    utility methods for serialization and access.
    """
    def __init__(self, 
                 from_agent_id: str,
                 to_agent_id: str,
                 **kwargs: Any) -> None:
        """Initialize a new event.
        
        Args:
            from_agent_id: The ID of the agent sending the event
            to_agent_id: The ID of the agent receiving the event
            **kwargs: Additional event attributes
        """
        # Core event identity fields
        self.event_id: str = kwargs.get("event_id", gen_id())
        self.timestamp: float = kwargs.get("timestamp", time.time())
        if isinstance(self.timestamp, datetime.datetime):
            self.timestamp=float(self.timestamp.timestamp())
        self.event_kind: str = self.__class__.__name__ if "event_kind" not in kwargs else kwargs["event_kind"]
        
        # Source and target identifiers
        self.from_agent_id: str = from_agent_id
        self.to_agent_id: str = to_agent_id
        self.parent_event_id: Optional[str] = kwargs.get("parent_event_id", None)
        
    def __str__(self) -> str:
        # 获取所有实例属性
        attrs = {key: value for key, value in vars(self).items()}
        # 转换为易读的字符串格式
        return f"{self.event_kind}({', '.join(f'{k}={repr(v)}' for k, v in attrs.items())})"

    def to_dict(self) -> dict:
        """Convert event to a dictionary format.
        
        Returns:
            dict: A dictionary containing all event attributes, with standard fields:
                - event_type: The type/kind of the event
                - event_id: Unique identifier for this event
                - source_id: The ID of the event sender
                - target_id: The ID of the event receiver
                - timestamp: Event creation timestamp
                - data: All other event attributes
        """
        # Get all instance variables
        all_attrs = vars(self)
        
        # Remove standard fields from data payload
        data = {k: v for k, v in all_attrs.items() 
               if k not in ['event_kind', 'from_agent_id', 'to_agent_id', 'event_id', 'timestamp'] and v is not None}
        
        # Construct the standard event format
        return {
            "event_type": self.event_kind,
            "event_id": self.event_id,
            "source_id": str(self.from_agent_id),
            "target_id": str(self.to_agent_id),
            "timestamp": self.timestamp,
            "data": data
        }
        
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get an attribute value by key with a default fallback.
        
        Args:
            key: The attribute name to retrieve
            default: Value to return if the attribute doesn't exist
            
        Returns:
            The attribute value if it exists, otherwise the default value
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class EndEvent(Event):
    """Event to signal agent termination."""
    def __init__(self,
                 from_agent_id: str, 
                 to_agent_id: str,
                 reason: str = "normal_termination",
                 **kwargs: Any) -> None:
        """Initialize a termination event.
        
        Args:
            from_agent_id: ID of the sender
            to_agent_id: ID of the receiver (use "all" for global termination)
            reason: Reason for termination
            **kwargs: Additional keyword arguments
        """
        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.reason = reason
        
class DataEvent(Event):
    """Event for data access across agents and environment."""
    def __init__(self,
                 from_agent_id: str,
                 to_agent_id: str,
                 source_type: str,  # "AGENT" or "ENV"
                 target_type: str,  # "AGENT" or "ENV" 
                 key: str,
                 default: Any = None,
                 **kwargs) -> None:
        """
        Initialize a data access event.
        
        Args:
            from_agent_id: ID of requesting entity
            to_agent_id: ID of entity that should receive request
            source_type: Type of requesting entity (AGENT/ENV)
            target_type: Type of target entity (AGENT/ENV)
            key: Data key to access
            default: Default value if key not found
            **kwargs: Additional keyword arguments
        """
        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.request_id = kwargs.get("request_id", gen_id())
        self.source_type = source_type
        self.target_type = target_type
        self.key = key
        self.default = default

class DataResponseEvent(Event):
    """Event for data access response."""
    def __init__(self,
                 from_agent_id: str,
                 to_agent_id: str,
                 request_id: str,
                 key: str,
                 data_value: Any = None,
                 success: bool = True,
                 error: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initialize a data response event.
        
        Args:
            from_agent_id: ID of responding entity
            to_agent_id: ID of requesting entity
            request_id: ID of the originating request
            key: Data key that was accessed
            data_value: Value of the data if success
            success: Whether the query was successful
            error: Error message if not successful
            **kwargs: Additional keyword arguments
        """
        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.request_id = request_id
        self.key = key
        self.data_value = data_value
        self.success = success
        self.error = error

class DataUpdateEvent(Event):
    """Event for updating data across agents and environment."""
    def __init__(self,
                 from_agent_id: str,
                 to_agent_id: str,
                 source_type: str,  # "AGENT" or "ENV"
                 target_type: str,  # "AGENT" or "ENV" 
                 key: str,
                 value: Any,
                 **kwargs) -> None:
        """
        Initialize a data update event.
        
        Args:
            from_agent_id: ID of requesting entity
            to_agent_id: ID of entity that should receive update
            source_type: Type of requesting entity (AGENT/ENV)
            target_type: Type of target entity (AGENT/ENV)
            key: Data key to update
            value: New value to set
            **kwargs: Additional keyword arguments
        """
        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.request_id = kwargs.get("request_id", gen_id())
        self.source_type = source_type
        self.target_type = target_type
        self.key = key
        self.value = value

class DataUpdateResponseEvent(Event):
    """Event for data update response."""
    def __init__(self,
                 from_agent_id: str,
                 to_agent_id: str,
                 request_id: str,
                 key: str,
                 success: bool = True,
                 error: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initialize a data update response event.
        
        Args:
            from_agent_id: ID of responding entity
            to_agent_id: ID of requesting entity
            request_id: ID of the originating request
            key: Data key that was updated
            success: Whether the update was successful
            error: Error message if not successful
            **kwargs: Additional keyword arguments
        """
        super().__init__(from_agent_id, to_agent_id, **kwargs)
        self.request_id = request_id
        self.key = key
        self.success = success
        self.error = error
    