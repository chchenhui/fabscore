"""
AI Scientist Meta-Layer for Autonomous Pharmaceutical Research
Implements the Meta-Scientist that generates hypotheses and conducts methodological research.
"""

__version__ = "1.0.0"
__author__ = "Claude-3.5-Sonnet-20241022"

# Core components
from .meta_scientist import MetaScientist
from .experiment_orchestrator import ExperimentOrchestrator  
from .evidence_grounding import EvidenceGroundingAgent, GroundedClaim
from .real_ai_parser import RealAIParser

# Multi-LLM integration
from .model_router import get_router, TaskType

# Data structures  
from .schemas import (
    ResearchHypothesis,
    ExperimentalProtocol, 
    ResearchResults,
    AuthorshipLedger
)