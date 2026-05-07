"""
This module provides parsers for extracting content between tags in model responses.
"""

from typing import Any, Optional, List, Dict, Union

from loguru import logger

from ..core.model_response import ModelResponse
from .base import ParserBase


class TagParser(ParserBase):
    """
    Parser for content surrounded by tags.
    
    This parser extracts content between specified start and end tags
    in model responses and provides it as parsed output.
    """
    
    def __init__(
        self,
        tag_start: str,
        tag_end: str,
        content_hint: Optional[str] = None
    ):
        """
        Initialize the parser.
        
        Args:
            tag_start: The start tag for the content block.
            tag_end: The end tag for the content block.
            content_hint: Optional hint for the expected content structure.
        """
        self.tag_start = tag_start
        self.tag_end = tag_end
        
        if content_hint is not None:
            self.content_hint = content_hint
        else:
            self.content_hint = "{your_content}"
            
    def parse(self, response: ModelResponse) -> ModelResponse:
        """
        Extract content between tags from the response.
        
        Args:
            response: The model response to parse.
            
        Returns:
            An updated ModelResponse with extracted content in the parsed field.
            
        Raises:
            ValueError: If content between tags cannot be extracted.
        """
        # Get the text content
        text = response.text
        
        if not text:
            raise ValueError("Response text is empty")
        
        # Find the start tag
        start_idx = text.find(self.tag_start)
        
        if start_idx == -1:
            raise ValueError(f"Start tag '{self.tag_start}' not found in response")
        
        # Find the end tag after the start tag
        content_start = start_idx + len(self.tag_start)
        end_idx = text.find(self.tag_end, content_start)
        
        if end_idx == -1:
            raise ValueError(f"End tag '{self.tag_end}' not found in response")
        
        # Extract the content between tags
        tag_content = text[content_start:end_idx].strip()
        
        response.parsed = tag_content
        return response
    
    @property
    def format_instruction(self) -> str:
        """
        Get the format instruction for the model.
        
        Returns:
            A string with instructions on the expected response format.
        """
        return (
            f"Respond with your content between the following tags:\n"
            f"{self.tag_start}\n{self.content_hint}\n{self.tag_end}"
        )


class MultiTagParser(ParserBase):
    """
    Parser for multiple tag-separated content blocks in a response.
    
    This parser extracts multiple content blocks between specified tags
    and provides them as a list or dictionary.
    """
    
    def __init__(
        self,
        tag_start: str,
        tag_end: str,
        named_tags: bool = False,
        required_tags: Optional[List[str]] = None,
        content_hint: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the parser.
        
        Args:
            tag_start: The start tag prefix for the content blocks.
            tag_end: The end tag for the content blocks.
            named_tags: Whether tags are named (e.g., <tag:name>) and should be parsed as a dict.
            required_tags: List of required tag names when named_tags is True.
            content_hint: Optional hint for the expected content structure.
        """
        self.tag_start = tag_start
        self.tag_end = tag_end
        self.named_tags = named_tags
        self.required_tags = required_tags or []
        self.content_hint = content_hint or {}
            
    def parse(self, response: ModelResponse) -> ModelResponse:
        """
        Extract multiple content blocks between tags from the response.
        
        Args:
            response: The model response to parse.
            
        Returns:
            An updated ModelResponse with extracted content blocks in the parsed field.
            
        Raises:
            ValueError: If content blocks cannot be extracted or required tags are missing.
        """
        # Get the text content
        text = response.text
        
        if not text:
            raise ValueError("Response text is empty")
        
        # Extract content
        if self.named_tags:
            parsed_content = self._extract_named_tags(text)
            
            # Check required tags
            missing_tags = [tag for tag in self.required_tags if tag not in parsed_content]
            if missing_tags:
                tags_str = ", ".join(missing_tags)
                raise ValueError(f"Missing required tags in response: {tags_str}")
        else:
            parsed_content = self._extract_unnamed_tags(text)
            
        response.parsed = parsed_content
        return response
    
    def _extract_named_tags(self, text: str) -> Dict[str, str]:
        """
        Extract content for named tags and return as a dictionary.
        
        Args:
            text: The text to parse.
            
        Returns:
            Dictionary with tag names as keys and extracted content as values.
        """
        result = {}
        pos = 0
        
        while True:
            # Find the next start tag
            start_idx = text.find(self.tag_start, pos)
            if start_idx == -1:
                break
                
            # Extract tag name
            tag_name_start = start_idx + len(self.tag_start)
            tag_name_end = text.find(">", tag_name_start)
            
            if tag_name_end == -1:
                pos = start_idx + 1
                continue
                
            tag_name = text[tag_name_start:tag_name_end].strip()
            
            # Find the content end
            content_start = tag_name_end + 1
            end_tag_pos = text.find(self.tag_end, content_start)
            
            if end_tag_pos == -1:
                pos = start_idx + 1
                continue
                
            # Extract content
            content = text[content_start:end_tag_pos].strip()
            result[tag_name] = content
            
            # Move position
            pos = end_tag_pos + len(self.tag_end)
            
        return result
    
    def _extract_unnamed_tags(self, text: str) -> List[str]:
        """
        Extract content for sequential tags and return as a list.
        
        Args:
            text: The text to parse.
            
        Returns:
            List of extracted content in the order they appear in the text.
        """
        result = []
        pos = 0
        
        while True:
            # Find the next start tag
            start_idx = text.find(self.tag_start, pos)
            if start_idx == -1:
                break
                
            # Find the content start and end
            content_start = start_idx + len(self.tag_start)
            end_tag_pos = text.find(self.tag_end, content_start)
            
            if end_tag_pos == -1:
                pos = start_idx + 1
                continue
                
            # Extract content
            content = text[content_start:end_tag_pos].strip()
            result.append(content)
            
            # Move position
            pos = end_tag_pos + len(self.tag_end)
            
        return result
    
    @property
    def format_instruction(self) -> str:
        """
        Get the format instruction for the model.
        
        Returns:
            A string with instructions on the expected response format.
        """
        if self.named_tags:
            examples = "\n".join([
                f"{self.tag_start}{tag}>{hint}\n{self.tag_end}"
                for tag, hint in self.content_hint.items()
            ]) or f"{self.tag_start}tag_name>your_content\n{self.tag_end}"
            
            required = ""
            if self.required_tags:
                required = f" Required tags: {', '.join(self.required_tags)}."
                
            return (
                f"Respond with your content in multiple named tags as follows:{required}\n"
                f"{examples}"
            )
        else:
            example = f"{self.tag_start}your_content\n{self.tag_end}"
            return (
                f"Respond with your content in multiple tags as follows:\n"
                f"{example}\n{example}"
            )
