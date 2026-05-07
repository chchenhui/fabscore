"""
Unified Reflection-Based Adjustments

This module handles dynamic rate and budget adjustments that emerge from agent reflections.
Unlike the previous separate system, adjustments are now part of the reflection process.
"""

import logging
from typing import Dict, Optional, List
from .entities import Freelancer, Client, Job

logger = logging.getLogger(__name__)


class ReflectionAdjustmentEngine:
    """Handles rate and budget adjustments from reflection responses."""
    
    def __init__(self, min_hourly_rate: float = 5.0, max_hourly_rate: float = 500.0,
                 min_budget_multiplier: float = 0.5, max_budget_multiplier: float = 2.0):
        """Initialize with sensible bounds for adjustments."""
        self.min_hourly_rate = min_hourly_rate
        self.max_hourly_rate = max_hourly_rate
        self.min_budget_multiplier = min_budget_multiplier
        self.max_budget_multiplier = max_budget_multiplier
    
    def apply_freelancer_rate_adjustment(
        self, 
        freelancer: Freelancer, 
        rate_adjustment: Dict, 
        round_num: int
    ) -> Optional[Dict]:
        """Apply rate adjustment from freelancer reflection."""
        
        change_type = rate_adjustment.get('change', 'keep').lower()
        if change_type == 'keep':
            return None
            
        old_rate = freelancer.min_hourly_rate
        new_rate = rate_adjustment.get('new_rate', old_rate)
        reasoning = rate_adjustment.get('reasoning', 'Strategic rate adjustment')
        
        # Validate and bound the new rate
        if new_rate < self.min_hourly_rate:
            new_rate = self.min_hourly_rate
        elif new_rate > self.max_hourly_rate:
            new_rate = self.max_hourly_rate
            
        # Check if change is significant enough (avoid tiny adjustments)
        change_factor = abs(new_rate - old_rate) / old_rate
        if change_factor < 0.05:  # Less than 5% change
            return None
            
        # Apply the adjustment
        freelancer.min_hourly_rate = new_rate
        freelancer.last_rate_adjustment_round = round_num
        
        logger.info(f"💰 {freelancer.name} adjusted rate: ${old_rate}/hr → ${new_rate}/hr")
        
        return {
            'old_rate': old_rate,
            'new_rate': new_rate,
            'change': change_type,
            'reasoning': reasoning
        }
    
    def apply_client_budget_adjustment(
        self, 
        client: Client, 
        budget_adjustment: Dict, 
        round_num: int
    ) -> Optional[Dict]:
        """Apply budget adjustment from client reflection."""
        
        change_type = budget_adjustment.get('change', 'keep').lower()
        if change_type == 'keep':
            return None
            
        old_multiplier = client.budget_multiplier
        reasoning = budget_adjustment.get('reasoning', 'Strategic budget adjustment')
        
        # Calculate new multiplier based on change type
        if change_type == 'increase':
            new_multiplier = old_multiplier * 1.15  # 15% increase
        elif change_type == 'decrease':
            new_multiplier = old_multiplier * 0.85  # 15% decrease
        else:
            return None
            
        # Validate and bound the multiplier
        if new_multiplier < self.min_budget_multiplier:
            new_multiplier = self.min_budget_multiplier
        elif new_multiplier > self.max_budget_multiplier:
            new_multiplier = self.max_budget_multiplier
            
        # Check if change is significant enough
        change_factor = abs(new_multiplier - old_multiplier) / old_multiplier
        if change_factor < 0.05:  # Less than 5% change
            return None
            
        # Apply the adjustment
        client.budget_multiplier = new_multiplier
        client.last_budget_adjustment_round = round_num
        
        logger.info(f"💼 {client.company_name} adjusted budget strategy: {old_multiplier:.1f}x → {new_multiplier:.1f}x")
        
        return {
            'old_multiplier': old_multiplier,
            'new_multiplier': new_multiplier,
            'change': change_type,
            'reasoning': reasoning
        }
    
    def adjust_unfilled_job_budgets(
        self, 
        client: Client, 
        unfilled_jobs: List[Job], 
        round_num: int,
        adjustment_percentage: float = 0.15
    ) -> List[Dict]:
        """
        Directly adjust the budget amounts of specific unfilled jobs.
        This is more realistic than just adjusting future job multipliers.
        """
        adjustments = []
        
        for job in unfilled_jobs:
            # Only adjust jobs that have been unfilled for at least 1 round
            if hasattr(job, 'rounds_unfilled') and job.rounds_unfilled >= 1:
                old_budget = job.budget_amount
                new_budget = old_budget * (1 + adjustment_percentage)
                
                # Apply the adjustment
                job.budget_amount = new_budget
                
                adjustment_info = {
                    'job_id': job.id,
                    'job_title': job.title,
                    'old_budget': old_budget,
                    'new_budget': new_budget,
                    'adjustment_percentage': adjustment_percentage,
                    'round': round_num,
                    'client_id': client.id
                }
                
                adjustments.append(adjustment_info)
                
                logger.info(f"💰 {client.company_name} increased budget for '{job.title}': ${old_budget:.0f} → ${new_budget:.0f} ({adjustment_percentage*100:.0f}% increase)")
        
        return adjustments


# Global instance for use throughout the marketplace
reflection_adjustment_engine = ReflectionAdjustmentEngine()
