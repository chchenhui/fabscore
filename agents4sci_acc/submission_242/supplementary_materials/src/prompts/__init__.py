"""
Organized collection of LLM prompts used throughout the marketplace simulation.
This module centralizes all prompts to make them easier to maintain and update.

Only exports functions that are actively used in the marketplace simulation.
"""

from .base import (
    generate_freelancer_persona_prompt,
    generate_client_persona_prompt
)
from .freelancer import (
    create_bidding_decision_prompt,
    generate_job_completion_prompt,
    generate_feedback_prompt
)
from .client import (
    create_gpt_job_posting_prompt,
    create_hiring_decision_prompt
)
from .reflection import (
    create_freelancer_reflection_prompt,
    create_client_reflection_prompt
)
from .validation import (
    validate_bidding_decision,
    validate_job_completion,
    validate_hiring_decision,
    validate_feedback,
    validate_reflection,
    validate_freelancer_personas,
    validate_client_personas,
    validate_job_posting
)

__all__ = [
    # Base prompts for agent creation
    'generate_freelancer_persona_prompt',
    'generate_client_persona_prompt',
    
    # Freelancer prompts  
    'create_bidding_decision_prompt',
    'generate_job_completion_prompt',
    'generate_feedback_prompt',
    
    # Client prompts
    'create_gpt_job_posting_prompt',
    'create_hiring_decision_prompt',
    
    # Reflection prompts
    'create_freelancer_reflection_prompt',
    'create_client_reflection_prompt',
    
    # Validation functions
    'validate_bidding_decision',
    'validate_job_completion', 
    'validate_hiring_decision',
    'validate_feedback',
    'validate_reflection',
    'validate_freelancer_personas',
    'validate_client_personas',
    'validate_job_posting',
]