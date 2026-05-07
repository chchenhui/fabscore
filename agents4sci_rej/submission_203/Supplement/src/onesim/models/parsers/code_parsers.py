"""
This module provides parsers for model responses, especially code responses.
"""

import re
from typing import Any, Dict, List, Optional, Union, Sequence

from loguru import logger
from pydantic import BaseModel

from ..core.model_response import ModelResponse

from .base import ParserBase


class CodeBlockParser(ParserBase):
    """
    Parser for code in markdown code blocks.
    
    This parser extracts code from markdown code blocks in
    model responses and makes it available for further processing.
    """
    
    def __init__(
        self,
        language: Optional[str] = None,
        tag_start: Optional[str] = None,
        tag_end: str = "```",
        content_hint: Optional[Any] = None
    ):
        """
        Initialize the parser.
        
        Args:
            language: The programming language to extract. If None, extracts any code block.
            tag_start: The start tag for the code block. If None, will use "```{language}".
            tag_end: The end tag for the code block.
            content_hint: Optional hint for the expected content structure.
        """
        self.language = language
        
        if tag_start is None and language is not None:
            self.tag_start = f"```{language}"
        else:
            self.tag_start = tag_start or "```"
            
        self.tag_end = tag_end
        
        if content_hint is not None:
            self.content_hint = content_hint
        else:
            self.content_hint = "{your_code_here}"
            
    def parse(self, response: ModelResponse) -> ModelResponse:
        """
        Extract code from the response.
        
        Args:
            response: The model response to parse.
            
        Returns:
            An updated ModelResponse with extracted code in the parsed field.
            
        Raises:
            ValueError: If code cannot be extracted.
        """
        # Get the text content
        text = response.text
        
        if not text:
            raise ValueError("Response text is empty")
        
        # If no specific language was requested, extract all code blocks
        if self.tag_start == "```":
            # Match any code block
            pattern = r"```(?:\w*)\n([\s\S]*?)```"
            matches = re.findall(pattern, text)
            
            if not matches:
                raise ValueError("No code blocks found in response")
            
            # Store all code blocks
            code_blocks = [block.strip() for block in matches]
            response.parsed = code_blocks
            return response
        
        # Find the specific code block
        start_idx = text.find(self.tag_start)
        
        if start_idx == -1:
            raise ValueError(f"Start tag '{self.tag_start}' not found in response")
        
        # Find the end tag after the start tag
        content_start = start_idx + len(self.tag_start)
        end_idx = text.find(self.tag_end, content_start)
        
        if end_idx == -1:
            raise ValueError(f"End tag '{self.tag_end}' not found in response")
        
        # Extract the code content
        code_content = text[content_start:end_idx].strip()
        
        response.parsed = code_content
        return response
    
    def parse_multiple(self, response: ModelResponse) -> ModelResponse:
        """
        Extract all instances of the specified code blocks from the response.
        
        Args:
            response: The model response to parse.
            
        Returns:
            An updated ModelResponse with a list of all extracted code blocks in the parsed field.
            
        Raises:
            ValueError: If no code blocks are found.
        """
        # Get the text content
        text = response.text
        
        if not text:
            raise ValueError("Response text is empty")
        
        # Determine the pattern based on whether a specific language is requested
        if self.language:
            pattern = f"```{self.language}\\n([\\s\\S]*?)```"
        else:
            pattern = r"```(?:\w*)\n([\s\S]*?)```"
            
        matches = re.findall(pattern, text)
        
        if not matches:
            raise ValueError(f"No {'code' if not self.language else self.language} blocks found in response")
        
        # Store all matched code blocks
        code_blocks = [block.strip() for block in matches]
        response.parsed = code_blocks
        return response
    
    @property
    def format_instruction(self) -> str:
        """
        Get the format instruction for the model.
        
        Returns:
            A string with instructions on the expected response format.
        """
        language_spec = self.language or "language"
        return (
            f"Respond with code in a markdown code block as follows:\n"
            f"```{language_spec}\n{self.content_hint}\n```"
        )


class LanguageCodeParser(CodeBlockParser):
    """
    A specialized version of CodeBlockParser that detects and extracts
    code blocks for specific programming languages.
    """
    
    def __init__(
        self,
        languages: Sequence[str],
        content_hint: Optional[Any] = None
    ):
        """
        Initialize the parser.
        
        Args:
            languages: A list of programming languages to extract (e.g., ["python", "javascript"]).
            content_hint: Optional hint for the expected content structure.
        """
        super().__init__(language=None, content_hint=content_hint)
        self.languages = languages
        
    def parse(self, response: ModelResponse) -> ModelResponse:
        """
        Extract code blocks for specified languages from the response.
        
        Args:
            response: The model response to parse.
            
        Returns:
            An updated ModelResponse with a dictionary mapping languages to 
            extracted code blocks in the parsed field.
            
        Raises:
            ValueError: If no code blocks for the specified languages are found.
        """
        # Get the text content
        text = response.text
        
        if not text:
            raise ValueError("Response text is empty")
        
        # Initialize results dictionary
        results = {}
        
        # Check for each language
        for lang in self.languages:
            pattern = f"```{lang}\\n([\\s\\S]*?)```"
            matches = re.findall(pattern, text)
            
            if matches:
                # Store all matched code blocks for this language
                results[lang] = [block.strip() for block in matches]
        
        if not results:
            raise ValueError(f"No code blocks for languages {self.languages} found in response")
        
        response.parsed = results
        return response
    
    @property
    def format_instruction(self) -> str:
        """
        Get the format instruction for the model.
        
        Returns:
            A string with instructions on the expected response format.
        """
        instructions = ["Respond with code in markdown code blocks for the following languages:"]
        
        for lang in self.languages:
            instructions.append(f"```{lang}\n{self.content_hint}\n```")
        
        return "\n".join(instructions)