from typing import Dict, Any, Optional

class AgentSchema:
    """Class to represent the schema for a specific type of agent."""
    def __init__(self, schema_config: Dict[str, Any]):
        """
        Initialize the schema with a configuration dictionary.
        
        Args:
            schema_config (Dict[str, Any]): Configuration defining fields, types, and privacy settings.
        """
        self.schema = schema_config

    def get_default_values(self) -> Dict[str, Any]:
        """Return a dictionary of default values based on the schema."""
        default_values = {}
        for field, settings in self.schema.items():
            default_values[field] = settings.get("default", None)
        return default_values

    def is_private(self, field: str) -> bool:
        """Check if a field is private based on the schema."""
        return self.schema.get(field, {}).get("private", False)

    def get_type(self, field: str) -> Optional[str]:
        """Return the type of a field."""
        return self.schema.get(field, {}).get("type")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the schema as a dictionary."""
        return self.schema
