from abc import ABC, abstractmethod
import json
import csv
from typing import Dict, Any, List
import os
from .schema import AgentSchema

class ProfileBase(ABC):
    """Base class for profile handling public and private fields with property access."""

    _version: int = 1

    def __init__(self, schema: AgentSchema = None, profile_data: Dict[str, Any] = None):
        """
        Initialize the profile with a JSON config file or a config dictionary.
        
        Args:
            schema (AgentSchema): Schema defining the profile structure
            profile_data (Dict[str, Any], optional): Directly provided profile data
        """
        self.schema = schema
        self._public_fields = {}
        self._private_fields = {}
        
        # Initialize profiles with default values
        self._initialize_profiles()
        
        # If profile data is provided, update the fields
        if profile_data:
            self._update_fields(profile_data)

    def _initialize_profiles(self):
        """Initialize profiles with default values from schema."""
        for key, value in self.schema.schema.items():
            default_value = value.get("default", None)
            if value.get("private", False):
                self._private_fields[key] = default_value
            else:
                self._public_fields[key] = default_value

    def _update_fields(self, data: Dict[str, Any]):
        """Update fields with provided data."""
        for key, value in data.items():
            if key in self._public_fields:
                self._public_fields[key] = value
            elif key in self._private_fields:
                self._private_fields[key] = value
            else:
                raise KeyError(f"Field '{key}' not found in the profile schema.")

    def __getattr__(self, name: str) -> Any:
        """
        Provide dynamic property-like access to profile fields.
        
        Args:
            name (str): Name of the field to access
            
        Returns:
            Any: Value of the requested field
            
        Raises:
            AttributeError: If the field doesn't exist in either public or private fields
        """
        if name in self._public_fields:
            return self._public_fields[name]
        elif name in self._private_fields:
            return self._private_fields[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """
        Provide dynamic property-like setting of profile fields.
        
        Args:
            name (str): Name of the field to set
            value (Any): Value to set
            
        Raises:
            AttributeError: If trying to set a field that doesn't exist in the schema
        """
        # Allow setting of class attributes defined in __init__
        if name in ['schema', '_public_fields', '_private_fields']:
            super().__setattr__(name, value)
            return

        if name in self._public_fields:
            self._public_fields[name] = value
        elif name in self._private_fields:
            self._private_fields[name] = value
        else:
            #super().__setattr__(name, value)
            self._public_fields[name] = value

    @staticmethod
    def load_profiles_from_csv(csv_file: str, agents: List['ProfileBase']):
        """Load profiles from a CSV file and initialize the agents' profiles."""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")

        with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row, agent in zip(reader, agents):
                agent._update_fields(row)

    @property
    def public_fields(self) -> Dict[str, Any]:
        """Get all public fields."""
        return self._public_fields.copy()

    @property
    def private_fields(self) -> Dict[str, Any]:
        """Get all private fields."""
        return self._private_fields.copy()

    def get_data(self, field_name: str) -> Any:
        """
        Get a field value by name.
        
        Args:
            field_name (str): Name of the field to retrieve
            
        Returns:
            Any: Value of the requested field
            
        Raises:
            KeyError: If the field doesn't exist
        """
        if field_name in self._public_fields:
            return self._public_fields[field_name]
        elif field_name in self._private_fields:
            return self._private_fields[field_name]
        raise KeyError(f"Field '{field_name}' not found in profile")
    
    def get(self,key:str,default:Any=None)->Any:
        """
        Get a field value by name.
        """
        if key in self._public_fields:
            return self._public_fields[key]
        elif key in self._private_fields:
            return self._private_fields[key]
        return default

    @abstractmethod
    def get_profile(self) -> Dict[str, Any]:
        """Return the profile as a dictionary."""
        pass

    @abstractmethod
    def generate_profile(self):
        """Generate the profile based on the schema."""
        pass

    @abstractmethod
    def get_profile_str(self) -> str:
        """Return a string representation of the profile."""
        pass