"""
Validation utilities for LLM responses using Pydantic models.
"""

import json
import logging
from typing import TypeVar, Type, Optional
from pydantic import BaseModel, ValidationError

from .models import (
    BiddingDecisionResponse,
    JobCompletionResponse,
    HiringDecisionResponse,
    FeedbackResponse,
    ReflectionResponse,
    FreelancerPersonaResponse,
    ClientPersonaResponse,
    JobPostingResponse
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

def validate_llm_response(response_text: str, model_class: Type[T]) -> Optional[T]:
    """
    Validate and parse LLM response using Pydantic models.
    
    Args:
        response_text: Raw JSON response from LLM
        model_class: Pydantic model class to validate against
    
    Returns:
        Validated model instance or None if validation fails
    """
    try:
        # Parse JSON
        response_data = json.loads(response_text)
        
        # Validate with Pydantic model
        validated_response = model_class(**response_data)
        
        return validated_response
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Response text: {response_text}")
        return None
        
    except ValidationError as e:
        logger.error(f"Response validation failed for {model_class.__name__}: {e}")
        logger.error(f"Response data: {response_data}")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error validating response: {e}")
        return None


def extract_json_from_response(content: str) -> Optional[str]:
    """Extract JSON from GPT response, handling markdown code blocks and arrays."""
    
    # Try to find JSON within markdown code blocks first
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end != -1:
            return content[start:end].strip()
    
    # Look for JSON arrays first (they start with [)
    array_start = content.find('[')
    array_end = content.rfind(']') + 1
    if array_start != -1 and array_end > array_start:
        return content[array_start:array_end]
    
    # Fallback to looking for JSON objects (they start with {)
    obj_start = content.find('{')
    obj_end = content.rfind('}') + 1
    if obj_start != -1 and obj_end > obj_start:
        return content[obj_start:obj_end]
    
    return None


# Convenience functions for each response type
def validate_bidding_decision(response_text: str) -> Optional[BiddingDecisionResponse]:
    """Validate bidding decision response."""
    return validate_llm_response(response_text, BiddingDecisionResponse)

def validate_job_completion(response_text: str) -> Optional[JobCompletionResponse]:
    """Validate job completion response.""" 
    return validate_llm_response(response_text, JobCompletionResponse)

def validate_hiring_decision(response_text: str) -> Optional[HiringDecisionResponse]:
    """Validate hiring decision response."""
    return validate_llm_response(response_text, HiringDecisionResponse)

def validate_feedback(response_text: str) -> Optional[FeedbackResponse]:
    """Validate feedback response."""
    return validate_llm_response(response_text, FeedbackResponse)

def validate_reflection(response_text: str) -> Optional[ReflectionResponse]:
    """Validate reflection response."""
    return validate_llm_response(response_text, ReflectionResponse)


def validate_freelancer_personas(response_content: str) -> Optional[list[FreelancerPersonaResponse]]:
    """Validate freelancer persona generation response (array)."""
    try:
        json_str = extract_json_from_response(response_content)
        if not json_str:
            return None
        data = json.loads(json_str)
        if not isinstance(data, list):
            logger.error("Expected array of freelancer personas")
            return None
        
        validated_personas = []
        for persona_data in data:
            try:
                persona = FreelancerPersonaResponse(**persona_data)
                validated_personas.append(persona)
            except ValidationError as e:
                logger.warning(f"Skipping invalid freelancer persona: {e}")
                continue
        
        return validated_personas if validated_personas else None
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to validate freelancer personas: {e}")
        return None


def validate_client_personas(response_content: str) -> Optional[list[ClientPersonaResponse]]:
    """Validate client persona generation response (array)."""
    try:
        json_str = extract_json_from_response(response_content)
        if not json_str:
            return None
        data = json.loads(json_str)
        if not isinstance(data, list):
            logger.error("Expected array of client personas")
            return None
        
        validated_personas = []
        for persona_data in data:
            try:
                persona = ClientPersonaResponse(**persona_data)
                validated_personas.append(persona)
            except ValidationError as e:
                logger.warning(f"Skipping invalid client persona: {e}")
                continue
        
        return validated_personas if validated_personas else None
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to validate client personas: {e}")
        return None


def validate_job_posting(response_content: str) -> Optional[JobPostingResponse]:
    """Validate job posting generation response."""
    return validate_llm_response(response_content, JobPostingResponse)
