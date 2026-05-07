"""
Job Ranking Algorithm Module

Simple, modular job ranking using semantic skill matching and configurable scoring.
"""

from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .entities import Freelancer, Job

from .scoring_components import SimplifiedRankingCalculator, ScoringConfig

logger = logging.getLogger(__name__)


class JobRankingCalculator:
    """
    Simple wrapper around the new modular ranking system.
    
    Provides backward compatibility while using the improved algorithm.
    """
    
    def __init__(self, relevance_mode: str = 'strict'):
        """
        Initialize the ranking calculator.
        
        Args:
            relevance_mode: One of 'strict', 'moderate', or 'relaxed'
        """
        self.relevance_mode = relevance_mode
        self.config = self._create_config_for_mode(relevance_mode)
        self.calculator = SimplifiedRankingCalculator(self.config)
    
    def calculate_job_relevance(self, freelancer: 'Freelancer', job: 'Job') -> float:
        """
        Calculate job relevance using the new simplified algorithm.
        
        Much cleaner than before:
        - Uses semantic skill matching instead of string matching
        - Modular scoring components
        - Configurable weights based on relevance mode
        - No complex category validation logic
        """
        return self.calculator.calculate_job_relevance(freelancer, job)
    
    def get_detailed_scores(self, freelancer: 'Freelancer', job: 'Job'):
        """Get detailed score breakdown for debugging"""
        return self.calculator.get_detailed_scores(freelancer, job)
    
    def _create_config_for_mode(self, relevance_mode: str) -> ScoringConfig:
        """Create scoring configuration based on relevance mode using pure ranking approach"""
        if relevance_mode == 'strict':
            # Strict mode: Heavy emphasis on skills, conservative budget tolerance
            return ScoringConfig(
                skill_weight=0.7,  # Skills matter most
                budget_weight=0.2,
                timeline_weight=0.1,
                budget_tolerance_percent=15.0,  # Conservative budget tolerance
                strict_budget=False
                # NO THRESHOLDS - let ranking decide!
            )
        elif relevance_mode == 'moderate':
            # Moderate mode: Balanced weighting
            return ScoringConfig(
                skill_weight=0.6,  # Balanced approach
                budget_weight=0.25,
                timeline_weight=0.15,
                budget_tolerance_percent=25.0,  # Standard tolerance
                strict_budget=False
                # NO THRESHOLDS - let ranking decide!
            )
        else:  # relaxed
            # Relaxed mode: More balanced, higher budget tolerance
            return ScoringConfig(
                skill_weight=0.5,  # Less skill emphasis
                budget_weight=0.3,  # Budget becomes more important
                timeline_weight=0.2,
                budget_tolerance_percent=50.0,  # High budget tolerance
                strict_budget=False
                # NO THRESHOLDS - let ranking decide!
            )

