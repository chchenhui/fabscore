"""Prompt configurations for different evaluation modes."""

from typing import Dict, Any
from enum import Enum


class PromptType(Enum):
    """Types of prompts available."""
    STEP_BY_STEP_BOXED = "step_by_step_boxed"
    DIRECT_ANSWER = "direct_answer"


PROMPT_CONFIGS: Dict[PromptType, Dict[str, Any]] = {
    PromptType.STEP_BY_STEP_BOXED: {
        "system_message": "You are a helpful assistant that solves arithmetic problems accurately.",
        "user_template": """Solve this arithmetic problem step by step and provide the final numerical answer in a box.

Problem: {problem}

Please show your work and end with \\boxed{{X}} where X is the numerical result.""",
        "answer_pattern": r"\\boxed\{([^}]+)\}",
        "description": "Step-by-step reasoning with boxed final answer"
    },
    
    PromptType.DIRECT_ANSWER: {
        "system_message": "You are a calculator that outputs only numerical results. Do not show work or explain your reasoning.",
        "user_template": """Calculate: {problem}

Output only the numerical answer, nothing else.""",
        "answer_pattern": r"^\s*(-?\d+(?:\.\d+)?)\s*$",
        "description": "Direct numerical answer only"
    }
}


def get_prompt_config(prompt_type: PromptType) -> Dict[str, Any]:
    """Get prompt configuration for a given type."""
    return PROMPT_CONFIGS[prompt_type]


def get_available_prompt_types() -> list:
    """Get list of available prompt types."""
    return [pt.value for pt in PromptType]