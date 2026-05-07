"""
Pydantic models for validating LLM responses in the marketplace simulation.
These models ensure type safety and consistent data structures.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class BiddingDecisionResponse(BaseModel):
    """Response model for freelancer bidding decisions."""
    decision: Literal["yes", "no"] = Field(description="Whether to bid on the job")
    reasoning: str = Field(description="Explanation for the decision")
    message: Optional[str] = Field(None, description="Pitch message to client (required if decision is 'yes')")

    def model_post_init(self, __context):
        """Validate that required fields are provided when bidding."""
        if self.decision == "yes":
            if self.message is None:
                raise ValueError("message is required when decision is 'yes'")


class JobCompletionResponse(BaseModel):
    """Response model for job completion evaluation."""
    success: bool = Field(description="Whether the job was completed successfully")


class HiringDecisionResponse(BaseModel):
    """Response model for client hiring decisions."""
    selected_bid_number: int = Field(ge=0, description="Number of chosen bid (1, 2, 3, etc.) or 0 for 'none'")
    selected_freelancer: str = Field(description="Name of chosen freelancer or 'none'")
    reasoning: str = Field(description="Explanation of hiring choice")
    confidence_level: Literal["high", "medium", "low"] = Field(description="Confidence in the decision")
    learnings: Optional[str] = Field(None, description="Key insight from this decision")
    future_implications: Optional[str] = Field(None, description="Brief note on strategic impact")


class FeedbackResponse(BaseModel):
    """Response model for bid feedback."""
    main_reason: str = Field(description="Primary reason for not selecting the bidder")
    pitch_feedback: str = Field(description="Feedback on their message/pitch")
    advice: str = Field(description="Specific advice for improving future bids")


class ReflectionResponse(BaseModel):
    """Response model for agent reflections (freelancers and clients)."""
    key_insights: List[str] = Field(description="Main insights from recent experiences")
    strategy_adjustments: List[str] = Field(description="Planned changes to strategy")
    market_observations: List[str] = Field(description="Observations about market conditions")

    # Optional fields - present based on agent type
    rate_adjustment: Optional[Dict] = Field(default=None, description="Freelancer rate adjustment decision")
    budget_adjustment: Optional[Dict] = Field(default=None, description="Client budget adjustment decision")
    job_budget_adjustments: Optional[List[Dict]] = Field(default=None, description="Client specific job budget adjustments")


class FreelancerPersonaResponse(BaseModel):
    """Response model for a single freelancer persona."""
    name: str = Field(description="Freelancer name")
    category: str = Field(description="Primary freelancer category/specialization")
    skills: List[str] = Field(description="List of freelancer skills")
    min_hourly_rate: float = Field(gt=0, description="Minimum hourly rate")
    personality: str = Field(description="Personality description")
    motivation: str = Field(description="Professional motivation")
    background: str = Field(description="Professional background")
    preferred_project_length: str = Field(description="Preferred project duration")


class ClientPersonaResponse(BaseModel):
    """Response model for a single client persona."""
    company_name: str = Field(description="Company name")
    company_size: str = Field(description="Company size category")
    budget_philosophy: str = Field(description="Budget approach philosophy")
    hiring_style: str = Field(description="Hiring decision style")
    background: str = Field(description="Company background and specialization")
    business_category: str = Field(description="Primary business category/domain")


class JobPostingResponse(BaseModel):
    """Response model for job posting generation."""
    title: str = Field(description="Job title")
    description: str = Field(description="Job description")
    skills_required: List[str] = Field(description="Required skills")
    budget_type: Literal["fixed", "hourly"] = Field(description="Budget type")
    budget_amount: float = Field(gt=0, description="Budget amount in USD")
    timeline: str = Field(description="Project timeline")
    special_requirements: List[str] = Field(description="Special requirements")
    category: str = Field(description="Job category")



