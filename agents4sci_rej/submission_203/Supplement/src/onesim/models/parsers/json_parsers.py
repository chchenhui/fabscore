"""
This module provides parsers for model responses, especially JSON responses.
"""

import json
from typing import Any, Dict, List, Optional, Union, Sequence

from loguru import logger
from pydantic import BaseModel

from ..core.model_response import ModelResponse

from .base import ParserBase


class JsonBlockParser(ParserBase):
    """
    Parser for JSON objects in code blocks.
    
    This parser extracts JSON objects from markdown code blocks in
    model responses and parses them into Python objects.
    """
    
    def __init__(
        self,
        tag_start: str = "```json",
        tag_end: str = "```",
        content_hint: Optional[Any] = None
    ):
        """
        Initialize the parser.
        
        Args:
            tag_start: The start tag for the JSON block.
            tag_end: The end tag for the JSON block.
            content_hint: Optional hint for the expected content structure.
        """
        self.tag_start = tag_start
        self.tag_end = tag_end
        
        if content_hint is not None:
            if isinstance(content_hint, str):
                self.content_hint = content_hint
            else:
                self.content_hint = json.dumps(
                    content_hint,
                    ensure_ascii=False,
                    indent=2
                )
        else:
            self.content_hint = "```json\n{your_json_object}\n```"
            
    def parse(self, response: ModelResponse) -> ModelResponse:
        """
        Extract and parse JSON from the response.
        
        Args:
            response: The model response to parse.
            
        Returns:
            An updated ModelResponse with parsed JSON in the parsed field.
            
        Raises:
            ValueError: If JSON cannot be extracted or parsed.
        """
        # Get the text content
        text = response.text
        
        if not text:
            raise ValueError("Response text is empty")
        
        # Find the JSON block
        start_idx = text.find(self.tag_start)
        
        if start_idx == -1:
            raise ValueError(f"Start tag '{self.tag_start}' not found in response")
        
        # Find the end tag after the start tag
        content_start = start_idx + len(self.tag_start)
        end_idx = text.find(self.tag_end, content_start)
        
        if end_idx == -1:
            raise ValueError(f"End tag '{self.tag_end}' not found in response")
        
        # Extract the JSON content
        json_content = text[content_start:end_idx].strip()
        
        # Parse the JSON
        try:
            parsed_json = json.loads(json_content)
            response.parsed = parsed_json
            return response
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    
    @property
    def format_instruction(self) -> str:
        """
        Get the format instruction for the model.
        
        Returns:
            A string with instructions on the expected response format.
        """
        return (
            f"Respond with a JSON object in a markdown code block as follows:\n"
            f"{self.tag_start}\n{self.content_hint}\n{self.tag_end}"
        )


class JsonDictParser(JsonBlockParser):
    """
    Parser for JSON dictionaries with field filtering capabilities.
    
    This parser extends JsonBlockParser to handle dictionaries specifically
    and provides methods to filter fields for different purposes.
    """
    
    def __init__(
        self,
        tag_start: str = "```json",
        tag_end: str = "```",
        content_hint: Optional[Any] = None,
        required_keys: Optional[List[str]] = None,
        keys_to_content: Union[str, bool, Sequence[str]] = True,
        keys_to_metadata: Union[str, bool, Sequence[str]] = False,
        schema: Optional[BaseModel] = None
    ):
        """
        Initialize the parser.
        
        Args:
            tag_start: The start tag for the JSON block.
            tag_end: The end tag for the JSON block.
            content_hint: Optional hint for the expected content structure.
            required_keys: List of keys that must be present in the parsed JSON.
            keys_to_content: Keys to include in content output.
            keys_to_metadata: Keys to include in metadata output.
            schema: Optional Pydantic model to validate the parsed JSON.
        """
        super().__init__(tag_start, tag_end, content_hint)
        
        self.required_keys = required_keys or []
        self.keys_to_content = keys_to_content
        self.keys_to_metadata = keys_to_metadata
        self.schema = schema
        
    def parse(self, response: ModelResponse) -> ModelResponse:
        """
        Parse and validate a JSON dictionary.
        
        Args:
            response: The model response to parse.
            
        Returns:
            An updated ModelResponse with parsed JSON dictionary in the parsed field.
            
        Raises:
            ValueError: If validation fails.
        """
        # First parse the JSON using parent class
        response = super().parse(response)
        
        # Ensure it's a dictionary
        if not isinstance(response.parsed, dict):
            raise ValueError(
                f"Expected a JSON dictionary, got {type(response.parsed).__name__}"
            )
        
        # Check required keys
        missing_keys = [key for key in self.required_keys if key not in response.parsed]
        if missing_keys:
            keys_str = ", ".join(missing_keys)
            raise ValueError(f"Missing required keys in response: {keys_str}")
        
        # Validate with schema if provided
        if self.schema is not None:
            try:
                validated_data = self.schema(**response.parsed).dict()
                response.parsed = validated_data
            except Exception as e:
                raise ValueError(f"Schema validation failed: {e}")
                
        return response
        
    def to_content(self, parsed_response: Dict) -> Union[str, Dict, None]:
        """
        Filter fields for content output.
        
        Args:
            parsed_response: The parsed dictionary to filter.
            
        Returns:
            Filtered content based on keys_to_content configuration.
        """
        return self._filter_by_keys(parsed_response, self.keys_to_content)
    
    def to_metadata(self, parsed_response: Dict) -> Union[str, Dict, None]:
        """
        Filter fields for metadata output.
        
        Args:
            parsed_response: The parsed dictionary to filter.
            
        Returns:
            Filtered metadata based on keys_to_metadata configuration.
        """
        return self._filter_by_keys(parsed_response, self.keys_to_metadata)
    
    def _filter_by_keys(
        self, 
        data: Dict, 
        keys: Union[str, bool, Sequence[str]]
    ) -> Union[str, Dict, None]:
        """
        Filter a dictionary by keys.
        
        Args:
            data: The dictionary to filter.
            keys: Key specification (True for all, False for none,
                  string for single key, sequence for multiple keys).
                  
        Returns:
            Filtered data based on keys configuration.
        """
        if isinstance(keys, bool):
            if keys:
                return data
            else:
                return None
                
        if isinstance(keys, str):
            return data.get(keys)
            
        # If keys is a sequence
        return {k: v for k, v in data.items() if k in keys} 