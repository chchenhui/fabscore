"""
Modular Scoring Components

Simple, composable components for different aspects of freelancer-job compatibility.
"""

import re
import logging
from typing import TYPE_CHECKING, Optional, Dict, Any
from dataclasses import dataclass

if TYPE_CHECKING:
    from .entities import Freelancer, Job

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for scoring weights and budget tolerance"""
    skill_weight: float = 0.6
    budget_weight: float = 0.25
    timeline_weight: float = 0.15
    
    # Budget tolerance settings
    budget_tolerance_percent: float = 20.0  # Allow up to 20% over budget
    strict_budget: bool = False  # If True, no tolerance for over-budget

class BudgetScorer:
    """Handles budget compatibility scoring"""
    
    @staticmethod
    def calculate_budget_score(freelancer: 'Freelancer', job: 'Job', config: ScoringConfig) -> float:
        """
        Calculate budget compatibility between freelancer rate and job budget.
        
        Returns:
            float: Score from 0.0 to 1.0
        """
        try:
            freelancer_rate = float(freelancer.min_hourly_rate)
            job_budget = BudgetScorer._extract_budget_amount(job.budget_amount)
            
            if job_budget is None:
                return 0.6  # Neutral score for unknown budgets
            
            if freelancer_rate <= job_budget:
                # Within budget - score based on how competitive the rate is
                if freelancer_rate <= job_budget * 0.8:
                    return 1.0  # Great value
                else:
                    return 0.9  # Good value
            else:
                # Over budget
                if config.strict_budget:
                    return 0.0
                
                # Check if within tolerance
                max_allowed = job_budget * (1 + config.budget_tolerance_percent / 100)
                if freelancer_rate <= max_allowed:
                    # Linearly decrease score based on how far over budget
                    excess_percent = (freelancer_rate - job_budget) / job_budget
                    tolerance_percent = config.budget_tolerance_percent / 100
                    penalty = excess_percent / tolerance_percent
                    return max(0.2, 0.6 - penalty * 0.4)  # Score from 0.6 down to 0.2
                else:
                    return 0.1  # Too expensive
                    
        except (ValueError, TypeError) as e:
            logger.exception(f"Error calculating budget score: {e}")
            return 0.5  # Neutral score on error
    
    @staticmethod
    def _extract_budget_amount(budget_text: str) -> Optional[float]:
        """Extract numeric budget from budget text"""
        if not budget_text:
            return None
            
        # Find all numbers in the budget text
        numbers = re.findall(r'\d+(?:\.\d+)?', str(budget_text))
        if not numbers:
            return None
        
        # Take the largest number (often the max budget)
        return max(float(num) for num in numbers)


class TimelineScorer:
    """Handles project timeline/availability scoring"""
    
    @staticmethod
    def calculate_timeline_score(freelancer: 'Freelancer', job: 'Job') -> float:
        """
        Calculate timeline compatibility between freelancer availability and job needs.
        
        Returns:
            float: Score from 0.0 to 1.0
        """
        # For now, use project length preferences
        # In a real system, this would consider:
        # - Freelancer availability calendar
        # - Job start date and deadline
        # - Freelancer workload
        
        freelancer_pref = freelancer.preferred_project_length.lower() if freelancer.preferred_project_length else ""
        job_desc = job.description.lower() if job.description else ""
        
        if not freelancer_pref or freelancer_pref == 'any':
            return 0.7  # Neutral positive score
        
        # Simple keyword matching for project length
        timeline_keywords = {
            'short': ['quick', 'fast', 'urgent', 'asap', 'immediate', 'short', 'small'],
            'medium': ['medium', 'moderate', 'standard', 'regular'],
            'long': ['long', 'extended', 'ongoing', 'large', 'complex', 'comprehensive']
        }
        
        job_timeline_type = None
        for timeline_type, keywords in timeline_keywords.items():
            if any(keyword in job_desc for keyword in keywords):
                job_timeline_type = timeline_type
                break
        
        if job_timeline_type is None:
            return 0.6  # Unknown timeline
        
        # Perfect match
        if freelancer_pref in job_timeline_type or job_timeline_type in freelancer_pref:
            return 1.0
        
        # Adjacent matches (short-medium, medium-long)
        adjacent_matches = {
            ('short', 'medium'): 0.7,
            ('medium', 'short'): 0.7,
            ('medium', 'long'): 0.7,
            ('long', 'medium'): 0.7,
        }
        
        for (pref, job_type), score in adjacent_matches.items():
            if pref in freelancer_pref and job_type == job_timeline_type:
                return score
        
        return 0.4  # Mismatched timeline preferences


# LocationScorer removed to avoid geographic bias


class SimplifiedRankingCalculator:
    """
    Simplified, modular ranking calculator.
    
    Much cleaner than the previous complex algorithm:
    - Uses semantic skill matching
    - Modular scoring components
    - Configurable weights
    - No hardcoded category logic
    """
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        from .skill_matcher import SemanticSkillMatcher
        
        self.config = config or ScoringConfig()
        self.skill_matcher = SemanticSkillMatcher()
        self.budget_scorer = BudgetScorer()
        self.timeline_scorer = TimelineScorer()
    
    def calculate_job_relevance(self, freelancer: 'Freelancer', job: 'Job') -> float:
        """
        Calculate overall job relevance score using pure ranking approach.
        
        Combines component scores with weights - NO THRESHOLDS!
        This allows for natural ranking and lets the market decide through competition.
        
        1. Calculate semantic skill match (with job description context)
        2. Calculate budget compatibility  
        3. Calculate timeline fit
        4. Combine with configurable weights
        
        Returns:
            float: Overall relevance score from 0.0 to 1.0
        """
        # Calculate component scores - now includes job description!
        skill_score = self.skill_matcher.calculate_skill_match_score(
            freelancer.skills, job.skills_required, job.description
        )
        
        budget_score = self.budget_scorer.calculate_budget_score(
            freelancer, job, self.config
        )
        
        timeline_score = self.timeline_scorer.calculate_timeline_score(
            freelancer, job
        )
        
        # Combine scores with weights - NO THRESHOLDS!
        # Let ranking and market competition determine job selection
        overall_score = (
            skill_score * self.config.skill_weight +
            budget_score * self.config.budget_weight +
            timeline_score * self.config.timeline_weight
        )
        
        return min(overall_score, 1.0)
    
    def get_detailed_scores(self, freelancer: 'Freelancer', job: 'Job') -> Dict[str, Any]:
        """Get detailed breakdown of scores for debugging/analysis"""
        skill_score = self.skill_matcher.calculate_skill_match_score(
            freelancer.skills, job.skills_required, job.description
        )
        
        budget_score = self.budget_scorer.calculate_budget_score(
            freelancer, job, self.config
        )
        
        timeline_score = self.timeline_scorer.calculate_timeline_score(
            freelancer, job
        )
        
        # Location scoring removed to avoid geographic bias
        
        overall_score = (
            skill_score * self.config.skill_weight +
            budget_score * self.config.budget_weight +
            timeline_score * self.config.timeline_weight
        )
        
        # Get text representations for debugging
        freelancer_profile = self.skill_matcher._create_freelancer_profile(freelancer.skills)
        job_context = self.skill_matcher._create_job_context(job.skills_required, job.description)
        
        return {
            'overall_score': min(overall_score, 1.0),
            'skill_score': skill_score,
            'budget_score': budget_score,
            'timeline_score': timeline_score,
            'freelancer_profile': freelancer_profile,
            'job_context': job_context,
            'weights': {
                'skill_weight': self.config.skill_weight,
                'budget_weight': self.config.budget_weight,
                'timeline_weight': self.config.timeline_weight,
            }
        }
