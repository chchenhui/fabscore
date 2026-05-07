"""
Core entities for the marketplace simulation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class Freelancer:
    """Represents a freelancer in the marketplace."""
    id: str
    name: str
    category: str
    skills: List[str]
    min_hourly_rate: float
    personality: str
    motivation: str
    background: str = ""
    preferred_project_length: str = "any"
    
    # Strategy tracking for adjustments
    last_rate_adjustment_round: int = 0
    
    # Performance metrics
    total_bids: int = 0
    total_hired: int = 0
    completed_jobs: int = 0
    bids_this_round: int = 0
    active_jobs: int = 0
    max_active_jobs: int = 3
    ongoing_jobs: List[Dict] = field(default_factory=list)
    job_history: List[Dict] = field(default_factory=list)
    bid_feedback: List[Dict] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate based on bids and hires."""
        return self.total_hired / self.total_bids if self.total_bids > 0 else 0
        
    @property
    def experience_level(self) -> str:
        """Determine experience level based on success rate."""
        if self.total_bids == 0:
            return "new"
        return "experienced" if self.success_rate >= 0.6 else "intermediate"
        
    @property
    def financial_situation(self) -> str:
        """Determine financial situation based on success rate."""
        if self.total_bids == 0:
            return "stable"
        return "good" if self.success_rate >= 0.8 else "stable"
        
    def can_bid(self, max_bids_per_round: int) -> bool:
        """Check if freelancer can bid based on limits."""
        return (self.bids_this_round < max_bids_per_round and 
                self.active_jobs < self.max_active_jobs)
        
    def add_bid(self) -> None:
        """Record a new bid."""
        self.total_bids += 1
        self.bids_this_round += 1
        
    def add_hire(self) -> None:
        """Record being hired."""
        self.total_hired += 1
        self.active_jobs += 1
        
    def complete_job(self, job_id: str, success: bool) -> None:
        """Record job completion."""
        if success:
            self.completed_jobs += 1
            
        self.active_jobs -= 1
        self.job_history.append({
            'job_id': job_id,
            'success': success,
            'completion_time': datetime.now()
        })
        
        # Memory optimization: Keep only recent job history (last 10 jobs)
        if len(self.job_history) > 10:
            self.job_history = self.job_history[-10:]
        
        # Remove from ongoing jobs
        self.ongoing_jobs = [j for j in self.ongoing_jobs if j['job_id'] != job_id]
        
    def add_feedback(self, feedback: Dict) -> None:
        """Add feedback from a bid."""
        if not hasattr(self, 'bid_feedback'):
            self.bid_feedback = []
        self.bid_feedback.append(feedback)
        
        # Memory optimization: Keep only recent feedback (last 10 items)
        if len(self.bid_feedback) > 10:
            self.bid_feedback = self.bid_feedback[-10:]
        
    def reset_round(self) -> None:
        """Reset per-round counters."""
        self.bids_this_round = 0
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'skills': self.skills,
            'min_hourly_rate': self.min_hourly_rate,
            'personality': self.personality,
            'motivation': self.motivation,
            'background': self.background,
            'preferred_project_length': self.preferred_project_length,
            'total_bids': self.total_bids,
            'total_hired': self.total_hired,
            'completed_jobs': self.completed_jobs,
            'bids_this_round': self.bids_this_round,
            'active_jobs': self.active_jobs,
            'max_active_jobs': self.max_active_jobs,
            'ongoing_jobs': self.ongoing_jobs,
            'job_history': self.job_history,
            'bid_feedback': self.bid_feedback
        }

@dataclass
class Client:
    """Represents a client company in the marketplace."""
    id: str
    company_name: str
    company_size: str
    budget_philosophy: str
    hiring_style: str
    background: str
    business_category: str  # The primary business category/domain
    next_job_round: int = 0
    
    # Strategy tracking for adjustments
    budget_multiplier: float = 1.0  # Multiplier for job budgets
    last_budget_adjustment_round: int = 0
    
    def can_post_job(self, current_round: int) -> bool:
        """Check if client can post a job this round."""
        return current_round >= self.next_job_round
        
    def set_next_job_round(self, current_round: int, cooldown: int) -> None:
        """Set the next round when client can post a job."""
        self.next_job_round = current_round + cooldown
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'company_name': self.company_name,
            'company_size': self.company_size,
            'budget_philosophy': self.budget_philosophy,
            'hiring_style': self.hiring_style,
            'background': self.background,
            'business_category': self.business_category,
            'next_job_round': self.next_job_round,
            'budget_multiplier': self.budget_multiplier,
            'last_budget_adjustment_round': self.last_budget_adjustment_round
        }

@dataclass
class Job:
    """Represents a job posting in the marketplace."""
    id: str
    client_id: str
    title: str
    description: str
    category: str
    skills_required: List[str]
    budget_type: str
    budget_amount: float
    timeline: str
    special_requirements: str = ""
    posted_time: datetime = field(default_factory=datetime.now)
    rounds_unfilled: int = 0  # Track how many rounds this job has been unfilled
    selected_freelancer: Optional[str] = None
    bids: List[Dict] = field(default_factory=list)
    
    def hire_freelancer(self, freelancer_id: str) -> None:
        """Hire a freelancer for this job."""
        self.selected_freelancer = freelancer_id
        
    def add_bid(self, bid: Dict) -> None:
        """Add a bid to this job."""
        self.bids.append(bid)
        
    def is_filled(self) -> bool:
        """Check if this job has been filled (has a selected freelancer)."""
        return self.selected_freelancer is not None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'client_id': self.client_id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'skills_required': self.skills_required,
            'budget_type': self.budget_type,
            'budget_amount': self.budget_amount,
            'timeline': self.timeline,
            'special_requirements': self.special_requirements,
            'posted_time': self.posted_time.isoformat(),
            'rounds_unfilled': self.rounds_unfilled,
            'selected_freelancer': self.selected_freelancer,
            'bids': self.bids
        }

@dataclass
class Bid:
    """Represents a bid on a job."""
    freelancer_id: str
    job_id: str
    proposed_rate: float
    message: str
    submission_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'freelancer_id': self.freelancer_id,
            'job_id': self.job_id,
            'proposed_rate': self.proposed_rate,
            'message': self.message,
            'submission_time': self.submission_time.isoformat()
        }

@dataclass
class HiringDecision:
    """Represents a hiring decision on a job."""
    job_id: str
    client_id: str
    selected_freelancer: Optional[str]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'client_id': self.client_id,
            'selected_freelancer': self.selected_freelancer,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }