"""
Shared utilities for market metrics calculation

This module contains shared functions for calculating market metrics to avoid duplication
between simulation (true_gpt_marketplace.py) and analysis (market_analysis.py) modules.
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def calculate_trend_direction(values: List[float], threshold: float = 0.1) -> str:
    """Calculate trend direction from a list of values
    
    Args:
        values: List of numeric values in chronological order
        threshold: Percentage threshold for determining significant change (0.1 = 10%)
    
    Returns:
        'improving', 'declining', or 'stable'
    """
    if len(values) < 2:
        return 'insufficient_data'
    
    # Calculate simple trend using first and last values
    if values[-1] > values[0] * (1 + threshold):  # threshold% increase
        return 'improving'
    elif values[-1] < values[0] * (1 - threshold):  # threshold% decrease
        return 'declining'
    else:
        return 'stable'

def calculate_health_grade(health_score: float) -> str:
    """Convert health score to categorical grade
    
    Args:
        health_score: Health score between 0.0 and 1.0
        
    Returns:
        Health grade string: 'excellent', 'good', 'fair', 'poor', or 'critical'
    """
    if health_score >= 0.75:
        return "excellent"
    elif health_score >= 0.60:
        return "good"
    elif health_score >= 0.40:
        return "fair"
    elif health_score >= 0.25:
        return "poor"
    else:
        return "critical"

def calculate_competitiveness_level(avg_bids_per_job: float) -> str:
    """Convert average bids per job to competitiveness level
    
    Args:
        avg_bids_per_job: Average number of bids received per job
        
    Returns:
        Competitiveness level: 'low', 'moderate', 'high', or 'very_high'
    """
    if avg_bids_per_job < 1.0:
        return "low"
    elif avg_bids_per_job < 2.5:
        return "moderate"
    elif avg_bids_per_job < 5.0:
        return "high"
    else:
        return "very_high"

def calculate_work_distribution_gini(work_counts: List[int]) -> float:
    """Calculate Gini coefficient for work distribution
    
    Args:
        work_counts: List of work counts (e.g., active jobs per freelancer)
        
    Returns:
        Gini coefficient between 0.0 (perfect equality) and 1.0 (perfect inequality)
    """
    if not work_counts or all(w == 0 for w in work_counts):
        return 0.0  # Perfect equality when no one has work
    
    # Sort the work counts
    work_counts = sorted(work_counts)
    n = len(work_counts)
    
    # Calculate Gini coefficient
    # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    total_work = sum(work_counts)
    if total_work == 0:
        return 0.0
    
    weighted_sum = sum((i + 1) * work for i, work in enumerate(work_counts))
    gini = (2 * weighted_sum) / (n * total_work) - (n + 1) / n
    
    return max(0.0, min(1.0, gini))  # Ensure it's between 0 and 1

def calculate_market_health_score(round_data: List[Dict[str, Any]]) -> float:
    """Calculate overall market health score using recent round data
    
    Args:
        round_data: List of round data dictionaries
        
    Returns:
        Health score between 0.0 and 1.0 (higher is better)
    """
    if not round_data:
        return 0.0
        
    health_factors = []
    recent_round = round_data[-1]
    
    # Factor 1: Fill rate (higher is better)
    jobs_posted = recent_round.get('jobs_posted', 0)
    jobs_filled = recent_round.get('jobs_filled_this_round', 0)
    fill_rate = jobs_filled / jobs_posted if jobs_posted > 0 else 0
    health_factors.append(fill_rate)
    
    # Factor 2: Balanced bid competition (optimal around 2-4 bids per job)
    avg_bids = recent_round.get('bid_distribution', {}).get('avg_bids_per_job', 0)
    if avg_bids == 0:
        bid_balance = 0.0
    elif avg_bids <= 1:
        bid_balance = avg_bids / 2.0  # Scale up to 0.5 max
    elif avg_bids <= 4:
        bid_balance = min(1.0, avg_bids / 3.0)  # Peak at 3 bids = 1.0
    else:
        bid_balance = max(0.2, 1.0 - (avg_bids - 4) / 6.0)  # Decline after 4
    health_factors.append(bid_balance)
    
    # Factor 3: Low rejection rate (lower is better)
    rejection_rate = recent_round.get('bid_rejection_metrics', {}).get('bid_rejection_rate', 0)
    rejection_health = 1.0 - rejection_rate  # Invert so lower rejection = higher health
    health_factors.append(rejection_health)
    
    # Factor 4: Good freelancer participation
    participation = recent_round.get('market_activity', {}).get('freelancer_participation_rate', 0)
    health_factors.append(participation)
    
    # Factor 5: Low saturation risk (if available)
    market_health = recent_round.get('market_health', {})
    if 'market_saturation_risk' in market_health:
        saturation_risk = market_health['market_saturation_risk']
        health_factors.append(1.0 - saturation_risk)
    
    return sum(health_factors) / len(health_factors) if health_factors else 0.0

def calculate_freelancer_tier(freelancer_data: Dict[str, Any]) -> str:
    """Calculate freelancer tier based on performance metrics
    
    Args:
        freelancer_data: Dictionary containing freelancer performance data
        
    Returns:
        Tier string: 'New', 'Established', 'Expert', or 'Elite'
    """
    hired = freelancer_data.get('total_hired', freelancer_data.get('total_jobs_hired', 0))
    completed = freelancer_data.get('completed_jobs', freelancer_data.get('successful_jobs', 0))
    completion_rate = completed / hired if hired > 0 else 0.0
    
    if hired < 3:
        return 'New'
    elif hired < 7:
        return 'Established' if completion_rate >= 0.6 else 'New'
    elif hired < 15:
        return 'Expert' if completion_rate >= 0.75 else 'Established'
    else:
        return 'Elite' if completion_rate >= 0.85 else 'Expert'

def calculate_client_tier(client_data: Dict[str, Any]) -> str:
    """Calculate client tier based on hiring performance
    
    Args:
        client_data: Dictionary containing client performance data
        
    Returns:
        Tier string: 'New', 'Established', 'Expert', or 'Elite'
    """
    jobs_posted = client_data.get('jobs_posted', client_data.get('total_jobs_posted', 0))
    successful_hires = client_data.get('successful_hires', client_data.get('total_successful_hires', 0))
    hire_success_rate = successful_hires / jobs_posted if jobs_posted > 0 else 0.0
    
    if jobs_posted == 0:
        return 'New'
    elif jobs_posted < 5:
        return 'New'
    elif jobs_posted < 20:
        return 'Established' if hire_success_rate >= 0.6 else 'New'
    elif jobs_posted < 50:
        return 'Expert' if hire_success_rate >= 0.75 else 'Established'
    else:
        return 'Elite' if hire_success_rate >= 0.85 else 'Expert'
