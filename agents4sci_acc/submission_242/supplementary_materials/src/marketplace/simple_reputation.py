"""
Simplified Reputation System for Marketplace Research

This module provides a streamlined reputation system focused on key marketplace dynamics:
- Job success rates and overall performance
- Category-specific expertise development  
- Simple tier progression
- Direct impact on marketplace opportunities

Simple reputation system focused on research-relevant mechanics.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from .market_metrics_utils import calculate_freelancer_tier, calculate_client_tier

logger = logging.getLogger(__name__)

class ReputationTier(Enum):
    """Simplified reputation tiers based on job completion performance"""
    NEW = "New"                    # 0-4 jobs hired
    ESTABLISHED = "Established"    # 5-14 jobs hired, >60% completion rate
    EXPERT = "Expert"              # 15+ jobs hired, >75% completion rate
    ELITE = "Elite"                # 40+ jobs hired, >85% completion rate

@dataclass
class FreelancerReputation:
    """Simplified freelancer reputation focused on marketplace impact"""
    # Core metrics
    overall_score: float           # 0.0-1.0 composite score
    job_completion_rate: float    # Job completion success rate (successful_jobs/total_jobs_hired)
    tier: ReputationTier          # Current reputation tier
    
    # Experience tracking
    total_jobs_hired: int         # Total jobs hired for (not bids!)
    successful_jobs: int          # Total successfully completed jobs
    total_bids: int              # Total bids submitted (for market activity tracking)
    total_earnings: float         # Total earnings (USD)
    
    # Category expertise (simplified)
    category_expertise: Dict[str, float]  # Category -> expertise level (0.0-1.0)
    
    # Temporal data
    join_date: datetime
    last_update: datetime

@dataclass 
class ClientReputation:
    """Simplified client reputation for hiring dynamics"""
    # Core metrics
    overall_score: float          # 0.0-1.0 composite score
    hire_success_rate: float      # Successful hires / job posts
    tier: ReputationTier          # Current reputation tier
    
    # Experience tracking
    total_jobs_posted: int        # Total jobs posted
    total_successful_hires: int   # Total successful hires
    total_spend: float            # Total spending (USD)
    
    # Behavioral patterns
    avg_budget_level: float       # Average job budget relative to market
    
    # Temporal data
    join_date: datetime
    last_update: datetime

class SimpleReputationManager:
    """Simplified reputation manager for marketplace research"""
    
    def __init__(self):
        self._freelancer_reps: Dict[str, FreelancerReputation] = {}
        self._client_reps: Dict[str, ClientReputation] = {}
        
    def initialize_freelancer(self, freelancer_id: str, skills: List[str] = None, freelancer_category: str = None) -> FreelancerReputation:
        """Initialize reputation for a new freelancer"""
        now = datetime.now()
        
        # Initialize category expertise based on freelancer's primary category
        category_expertise = {}
        if freelancer_category:
            # Set high expertise in freelancer's primary category
            category_expertise[freelancer_category] = 0.8  # Strong expertise in their field
        
        reputation = FreelancerReputation(
            overall_score=0.0,
            job_completion_rate=0.0,
            tier=ReputationTier.NEW,
            total_jobs_hired=0,
            successful_jobs=0,
            total_bids=0,
            total_earnings=0.0,
            category_expertise=category_expertise,
            join_date=now,
            last_update=now
        )
        
        self._freelancer_reps[freelancer_id] = reputation
        logger.info(f"✅ Initialized reputation for freelancer {freelancer_id}. Total freelancers: {len(self._freelancer_reps)}")
        return reputation
    
    def initialize_client(self, client_id: str) -> ClientReputation:
        """Initialize reputation for a new client"""
        now = datetime.now()
        
        reputation = ClientReputation(
            overall_score=0.0,
            hire_success_rate=0.0,
            tier=ReputationTier.NEW,
            total_jobs_posted=0,
            total_successful_hires=0,
            total_spend=0.0,
            avg_budget_level=1.0,  # Start at market average
            join_date=now,
            last_update=now
        )
        
        self._client_reps[client_id] = reputation
        logger.info(f"✅ Initialized reputation for client {client_id}. Total clients: {len(self._client_reps)}")
        return reputation
    
    def get_freelancer_reputation(self, freelancer_id: str) -> Optional[FreelancerReputation]:
        """Get freelancer reputation"""
        return self._freelancer_reps.get(freelancer_id)
    
    def get_client_reputation(self, client_id: str) -> Optional[ClientReputation]:
        """Get client reputation"""
        return self._client_reps.get(client_id)
    
    def update_freelancer_after_bid(self, freelancer_id: str, was_hired: bool, job_category: str = None):
        """Update freelancer reputation after a bid result"""
        reputation = self._freelancer_reps.get(freelancer_id)
        if not reputation:
            logger.warning(f"No reputation found for freelancer {freelancer_id}")
            return
        
        # Update bid count (for market activity tracking)
        reputation.total_bids += 1
        
        # Update hire metrics if hired (this is when they get the job, not complete it)
        if was_hired:
            reputation.total_jobs_hired += 1
            
            # Small category expertise increase for being hired (shows market confidence)
            if job_category and job_category in reputation.category_expertise:
                current_expertise = reputation.category_expertise[job_category]
                expertise_gain = 0.05 * (1 - current_expertise)  # Smaller gain for just being hired
                reputation.category_expertise[job_category] = min(1.0, current_expertise + expertise_gain)
        
        # Recalculate derived metrics
        self._update_freelancer_derived_metrics(reputation)
        reputation.last_update = datetime.now()
        
        logger.debug(f"Updated freelancer {freelancer_id} reputation after {'hire' if was_hired else 'bid'}: {reputation.overall_score:.2f}")
    
    def update_freelancer_after_job(self, freelancer_id: str, earnings: float, success: bool, job_category: str = None):
        """Update freelancer reputation after job completion"""
        reputation = self._freelancer_reps.get(freelancer_id)
        if not reputation:
            return
        
        # Update earnings
        reputation.total_earnings += earnings
        
        # Track successful job completion (this is what matters for reputation!)
        if success:
            reputation.successful_jobs += 1
        
        # Update category expertise based on completion success
        if job_category and job_category in reputation.category_expertise:
            current_expertise = reputation.category_expertise[job_category]
            if success:
                # Larger gain for successful job completion
                expertise_gain = 0.15 * (1 - current_expertise)
                reputation.category_expertise[job_category] = min(1.0, current_expertise + expertise_gain)
            else:
                # Small penalty for unsuccessful completion
                expertise_penalty = 0.05 * current_expertise
                reputation.category_expertise[job_category] = max(0.0, current_expertise - expertise_penalty)
        
        # Recalculate derived metrics
        self._update_freelancer_derived_metrics(reputation)
        reputation.last_update = datetime.now()
        
        logger.debug(f"Updated freelancer {freelancer_id} after job {'completion' if success else 'failure'}: completion rate {reputation.job_completion_rate:.2f}")
    
    def update_client_after_post(self, client_id: str, was_filled: bool, job_budget: float):
        """Update client reputation after job posting result"""
        reputation = self._client_reps.get(client_id)
        if not reputation:
            logger.warning(f"No reputation found for client {client_id}")
            return
        
        # Update posting metrics
        reputation.total_jobs_posted += 1
        if was_filled:
            reputation.total_successful_hires += 1
            reputation.total_spend += job_budget
        
        # Update average budget (moving average)
        alpha = 0.2
        reputation.avg_budget_level = (1 - alpha) * reputation.avg_budget_level + alpha * (job_budget / 50.0)  # Normalize by typical budget
        
        # Recalculate derived metrics
        self._update_client_derived_metrics(reputation)
        reputation.last_update = datetime.now()
        
        logger.debug(f"Updated client {client_id} reputation: {reputation.overall_score:.2f}")
    
    def _update_freelancer_derived_metrics(self, reputation: FreelancerReputation):
        """Update derived metrics for freelancer"""
        # Calculate job completion rate (successful jobs / total jobs hired)
        if reputation.total_jobs_hired > 0:
            reputation.job_completion_rate = reputation.successful_jobs / reputation.total_jobs_hired
        else:
            reputation.job_completion_rate = 0.0
        
        # Calculate overall score (weighted combination)
        base_score = reputation.job_completion_rate
        
        # Experience bonus (logarithmic scaling based on total jobs hired)
        import math
        experience_bonus = min(0.2, math.log(reputation.total_jobs_hired + 1) / 20)
        
        # Category expertise bonus (average of top 3 categories)
        if reputation.category_expertise:
            top_expertise = sorted(reputation.category_expertise.values(), reverse=True)[:3]
            expertise_bonus = sum(top_expertise) / len(top_expertise) * 0.1 if top_expertise else 0
        else:
            expertise_bonus = 0
        
        reputation.overall_score = min(1.0, base_score + experience_bonus + expertise_bonus)
        
        # Update tier based on performance
        reputation.tier = self._calculate_freelancer_tier(reputation)
    
    def _update_client_derived_metrics(self, reputation: ClientReputation):
        """Update derived metrics for client"""
        # Calculate hire success rate
        if reputation.total_jobs_posted > 0:
            reputation.hire_success_rate = reputation.total_successful_hires / reputation.total_jobs_posted
        else:
            reputation.hire_success_rate = 0.0
        
        # Calculate overall score
        base_score = reputation.hire_success_rate
        
        # Budget reasonableness bonus (closer to 1.0 = market-appropriate budgets)
        budget_bonus = 0.1 * (1 - abs(reputation.avg_budget_level - 1.0))
        
        reputation.overall_score = min(1.0, base_score + budget_bonus)
        
        # Update tier
        reputation.tier = self._calculate_client_tier(reputation)
    
    def _calculate_freelancer_tier(self, reputation: FreelancerReputation) -> ReputationTier:
        """Calculate freelancer tier based on job completion performance"""
        # Use shared tier calculation function
        freelancer_data = {
            'total_hired': reputation.total_jobs_hired,
            'completed_jobs': reputation.successful_jobs
        }
        tier_string = calculate_freelancer_tier(freelancer_data)
        return ReputationTier(tier_string)
    
    def _calculate_client_tier(self, reputation: ClientReputation) -> ReputationTier:
        """Calculate client tier based on hiring success"""
        # Use shared tier calculation function
        client_data = {
            'total_jobs_posted': reputation.total_jobs_posted,
            'total_successful_hires': reputation.total_successful_hires
        }
        tier_string = calculate_client_tier(client_data)
        return ReputationTier(tier_string)

# Global instance
simple_reputation_manager = SimpleReputationManager()
