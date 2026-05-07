"""
Market Dynamics Module

This module handles category-specific market dynamics, competition levels,
and difficulty factors for the simulated marketplace.

Key features:
- Dynamic rate adjustments based on supply/demand
- Category-specific competition factors
- Skill requirement scaling
- Market difficulty modifiers
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from src.marketplace.job_categories import JobCategory, category_manager
import numpy as np

@dataclass
class MarketState:
    """Current state of the market for a specific category"""
    category: JobCategory
    active_jobs: int
    active_freelancers: int
    avg_bid_rate: float
    success_rate: float
    competition_level: float
    last_updated: datetime

@dataclass
class CategoryDynamics:
    """Dynamic factors affecting a job category"""
    base_difficulty: float  # Base difficulty level (0-1)
    skill_importance: float  # How much skills matter (0-1)
    rate_elasticity: float  # How sensitive rates are to market conditions (0-1)
    entry_barrier: float  # Difficulty for new freelancers (0-1)
    specialization_bonus: float  # Bonus for specialized skills (0-1)

class MarketDynamicsManager:
    """Manages market dynamics and difficulty levels"""
    
    def __init__(self):
        self._market_states: Dict[JobCategory, MarketState] = {}
        self._category_dynamics = self._initialize_category_dynamics()
        self._update_interval = timedelta(hours=1)  # How often to update market states
    
    def get_market_state(self, category: JobCategory) -> MarketState:
        """Get current market state for a category"""
        if category not in self._market_states or \
           datetime.now() - self._market_states[category].last_updated > self._update_interval:
            self._update_market_state(category)
        return self._market_states[category]
    
    def calculate_job_difficulty(self, job: Dict, category: JobCategory) -> float:
        """Calculate difficulty level for a specific job (0-1 scale)"""
        category_def = category_manager.get_category(category)
        dynamics = self._category_dynamics[category]
        
        # Base difficulty from category
        difficulty = dynamics.base_difficulty
        
        # Adjust for required skills
        required_skills = set(s.lower() for s in job.get('skills_required', []))
        core_skills = set(s.lower() for s in category_def.core_skills)
        skill_overlap = len(required_skills & core_skills)
        skill_factor = min(1.0, skill_overlap / max(1, len(required_skills)))
        difficulty += (1 - skill_factor) * dynamics.skill_importance
        
        # Adjust for budget expectations
        budget = float(job.get('budget_amount', 0))
        avg_rate = category_def.metrics.avg_hourly_rate
        if budget < avg_rate * 0.8:  # Below market rate
            difficulty += 0.2
        elif budget > avg_rate * 1.5:  # Premium rate
            difficulty -= 0.1
        
        # Adjust for market conditions
        market_state = self.get_market_state(category)
        if market_state.competition_level > 0.7:  # High competition
            difficulty += 0.15
        
        return min(1.0, max(0.1, difficulty))
    
    def calculate_success_probability(
        self,
        freelancer: Dict,
        job: Dict,
        category: JobCategory,
        proposed_rate: float
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate probability of success for a bid (0-1 scale)"""
        category_def = category_manager.get_category(category)
        dynamics = self._category_dynamics[category]
        market_state = self.get_market_state(category)
        
        factors = {}
        
        # 1. Category match (30%)
        freelancer_category = freelancer.get('category', '')
        if freelancer_category == category.value[0]:
            category_match = 1.0  # Perfect match for same category
        else:
            category_match = 0.1  # Very low match for different categories
        factors['skill_match'] = category_match * 0.3
        
        # 2. Rate competitiveness (25%)
        rate_factor = self._calculate_rate_competitiveness(
            proposed_rate,
            category_def.metrics.avg_hourly_rate,
            market_state.avg_bid_rate,
            dynamics.rate_elasticity
        )
        factors['rate_competitiveness'] = rate_factor * 0.25
        
        # 3. Experience and reputation (20%)
        exp_factor = self._calculate_experience_factor(
            freelancer,
            category,
            dynamics.entry_barrier
        )
        factors['experience'] = exp_factor * 0.2
        
        # 4. Market conditions (15%)
        market_factor = self._calculate_market_factor(
            market_state.competition_level,
            market_state.success_rate
        )
        factors['market_conditions'] = market_factor * 0.15
        
        # 5. Specialization bonus (10%)
        spec_bonus = self._calculate_specialization_bonus(
            freelancer,
            category,
            dynamics.specialization_bonus
        )
        factors['specialization'] = spec_bonus * 0.1
        
        # Calculate total probability
        total_prob = sum(factors.values())
        return min(1.0, max(0.1, total_prob)), factors
    
    def update_market_metrics(
        self,
        category: JobCategory,
        new_job: Optional[Dict] = None,
        new_bid: Optional[Dict] = None,
        hire_result: Optional[Dict] = None
    ) -> None:
        """Update market metrics based on new activity"""
        if category not in self._market_states:
            self._initialize_market_state(category)
        
        state = self._market_states[category]
        
        if new_job:
            state.active_jobs += 1
        
        if new_bid:
            # Update average bid rate
            old_avg = state.avg_bid_rate
            bid_rate = float(new_bid.get('proposed_rate', 0))
            state.avg_bid_rate = (old_avg * state.active_jobs + bid_rate) / (state.active_jobs + 1)
        
        if hire_result:
            if hire_result.get('selected_freelancer'):
                state.success_rate = (state.success_rate * 9 + 1) / 10  # Rolling average
                state.active_jobs -= 1
            else:
                state.success_rate = (state.success_rate * 9) / 10
        
        # Update competition level
        if state.active_jobs > 0:
            state.competition_level = min(1.0, state.active_freelancers / (state.active_jobs * 2))
        
        state.last_updated = datetime.now()
    
    def _initialize_category_dynamics(self) -> Dict[JobCategory, CategoryDynamics]:
        """Initialize dynamics factors for each category"""
        return {
            JobCategory.ACCOUNTING: CategoryDynamics(
                base_difficulty=0.6,
                skill_importance=0.8,
                rate_elasticity=0.4,
                entry_barrier=0.7,
                specialization_bonus=0.6
            ),
            JobCategory.ADMIN: CategoryDynamics(
                base_difficulty=0.3,
                skill_importance=0.5,
                rate_elasticity=0.7,
                entry_barrier=0.3,
                specialization_bonus=0.4
            ),
            JobCategory.CUSTOMER_SERVICE: CategoryDynamics(
                base_difficulty=0.4,
                skill_importance=0.6,
                rate_elasticity=0.6,
                entry_barrier=0.4,
                specialization_bonus=0.3
            ),
            JobCategory.DATA_SCIENCE: CategoryDynamics(
                base_difficulty=0.8,
                skill_importance=0.9,
                rate_elasticity=0.5,
                entry_barrier=0.8,
                specialization_bonus=0.7
            ),
            JobCategory.DESIGN: CategoryDynamics(
                base_difficulty=0.5,
                skill_importance=0.7,
                rate_elasticity=0.6,
                entry_barrier=0.5,
                specialization_bonus=0.8
            ),
            JobCategory.ENGINEERING: CategoryDynamics(
                base_difficulty=0.7,
                skill_importance=0.8,
                rate_elasticity=0.5,
                entry_barrier=0.7,
                specialization_bonus=0.7
            ),
            JobCategory.IT: CategoryDynamics(
                base_difficulty=0.6,
                skill_importance=0.8,
                rate_elasticity=0.5,
                entry_barrier=0.6,
                specialization_bonus=0.7
            ),
            JobCategory.LEGAL: CategoryDynamics(
                base_difficulty=0.8,
                skill_importance=0.9,
                rate_elasticity=0.3,
                entry_barrier=0.9,
                specialization_bonus=0.8
            ),
            JobCategory.SALES: CategoryDynamics(
                base_difficulty=0.5,
                skill_importance=0.6,
                rate_elasticity=0.7,
                entry_barrier=0.4,
                specialization_bonus=0.5
            ),
            JobCategory.TRANSLATION: CategoryDynamics(
                base_difficulty=0.5,
                skill_importance=0.8,
                rate_elasticity=0.5,
                entry_barrier=0.6,
                specialization_bonus=0.7
            ),
            JobCategory.SOFTWARE: CategoryDynamics(
                base_difficulty=0.7,
                skill_importance=0.8,
                rate_elasticity=0.6,
                entry_barrier=0.6,
                specialization_bonus=0.7
            ),
            JobCategory.WRITING: CategoryDynamics(
                base_difficulty=0.4,
                skill_importance=0.7,
                rate_elasticity=0.6,
                entry_barrier=0.4,
                specialization_bonus=0.6
            )
        }
    
    def _initialize_market_state(self, category: JobCategory) -> None:
        """Initialize market state for a category"""
        category_def = category_manager.get_category(category)
        self._market_states[category] = MarketState(
            category=category,
            active_jobs=0,
            active_freelancers=0,
            avg_bid_rate=category_def.metrics.avg_hourly_rate,
            success_rate=0.5,  # Start at neutral
            competition_level=0.5,  # Start at neutral
            last_updated=datetime.now()
        )
    
    def _update_market_state(self, category: JobCategory) -> None:
        """Update market state based on recent activity"""
        if category not in self._market_states:
            self._initialize_market_state(category)
        else:
            # Decay values slightly over time
            state = self._market_states[category]
            time_factor = min(1.0, (datetime.now() - state.last_updated).total_seconds() / 3600)
            
            # Slowly return to neutral values
            state.competition_level = state.competition_level * (1 - 0.1 * time_factor) + 0.5 * 0.1 * time_factor
            state.success_rate = state.success_rate * (1 - 0.1 * time_factor) + 0.5 * 0.1 * time_factor
            
            # Update timestamp
            state.last_updated = datetime.now()
    
    def _calculate_rate_competitiveness(
        self,
        proposed_rate: float,
        category_avg_rate: float,
        market_avg_rate: float,
        elasticity: float
    ) -> float:
        """Calculate how competitive a proposed rate is"""
        # Compare to both category average and current market average
        category_factor = min(1.0, category_avg_rate / max(proposed_rate, 1))
        market_factor = min(1.0, market_avg_rate / max(proposed_rate, 1))
        
        # Weight factors based on rate elasticity
        return (category_factor * (1 - elasticity) + market_factor * elasticity)
    
    def _calculate_experience_factor(
        self,
        freelancer: Dict,
        category: JobCategory,
        entry_barrier: float
    ) -> float:
        """Calculate experience and reputation factor"""
        total_jobs = freelancer.get('total_hired', 0)
        success_rate = (freelancer.get('total_hired', 0) / 
                       max(freelancer.get('total_bids', 1), 1))
        
        # Experience curve with diminishing returns
        exp_score = min(1.0, np.log(total_jobs + 1) / np.log(50))  # Log scale up to 50 jobs
        
        # Combine with success rate
        combined_score = (exp_score + success_rate) / 2
        
        # Apply entry barrier
        if total_jobs == 0:  # New freelancer
            combined_score *= (1 - entry_barrier)
        
        return combined_score
    
    def _calculate_market_factor(
        self,
        competition_level: float,
        success_rate: float
    ) -> float:
        """Calculate market conditions factor"""
        # Higher competition reduces chances
        competition_factor = 1 - competition_level
        
        # Higher market success rate improves chances
        success_factor = success_rate
        
        return (competition_factor + success_factor) / 2
    
    def _calculate_specialization_bonus(
        self,
        freelancer: Dict,
        category: JobCategory,
        bonus_factor: float
    ) -> float:
        """Calculate specialization bonus based on category match and certifications"""
        # Calculate category match
        freelancer_category = freelancer.get('category', '')
        if freelancer_category == category.value[0]:
            expertise_match = 1.0  # Perfect match for same category
        else:
            expertise_match = 0.1  # Very low match for different categories
        
        # Check certification value for category
        category_def = category_manager.get_category(category)
        cert_importance = category_def.certification_value
        has_certs = bool(freelancer.get('certifications', []))
        cert_bonus = cert_importance if has_certs else 0.0
        
        # Combine expertise and certification bonuses
        return min(1.0, expertise_match * bonus_factor + cert_bonus * 0.3)

# Global instance for easy access
market_dynamics = MarketDynamicsManager()
