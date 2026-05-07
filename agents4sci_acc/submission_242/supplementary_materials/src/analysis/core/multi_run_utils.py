"""
Multi-Run Analysis Utilities

Shared utilities for statistical analysis and metrics extraction across 
multiple simulation runs, integrated with the existing analysis framework.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class SimulationMetrics:
    """Container for extracted simulation metrics"""
    fill_rate: float
    total_jobs: int
    total_bids: int
    bid_efficiency: float
    avg_bids_per_job: float
    participation_rate: float
    rejection_rate: float
    freelancer_hiring_rate: float
    gini_coefficient: float
    market_health_score: float


class MultiRunMetricsExtractor:
    """Utility class for extracting metrics from simulation data"""
    
    @staticmethod
    def extract_basic_metrics(simulation_data: Dict[str, Any]) -> SimulationMetrics:
        """Extract basic metrics from simulation data"""
        round_data = simulation_data.get('round_data', [])
        freelancer_profiles = simulation_data.get('freelancer_profiles', {})
        
        # Calculate basic metrics
        total_jobs = sum(r.get('jobs_posted', 0) for r in round_data)
        total_bids = sum(r.get('total_bids', 0) for r in round_data)
        total_filled = round_data[-1].get('jobs_filled', 0) if round_data else 0
        
        # Calculate participation and rejection rates
        participation_rates = [
            r.get('market_activity', {}).get('freelancer_participation_rate', 0) 
            for r in round_data
        ]
        rejection_rates = [
            r.get('bid_rejection_metrics', {}).get('bid_rejection_rate', 0) 
            for r in round_data
        ]
        
        # Calculate freelancer success
        successful_freelancers = sum(
            1 for f in freelancer_profiles.values() 
            if f.get('total_hired', 0) > 0
        )
        total_freelancers = len(freelancer_profiles)
        
        # Calculate Gini coefficient for work distribution
        work_distribution = [f.get('total_hired', 0) for f in freelancer_profiles.values()]
        gini = MultiRunMetricsExtractor.calculate_gini_coefficient(work_distribution)
        
        # Market health score (from last round)
        market_health = (
            round_data[-1].get('market_health', {}).get('health_score', 0) 
            if round_data else 0
        )
        
        return SimulationMetrics(
            fill_rate=total_filled / total_jobs if total_jobs > 0 else 0,
            total_jobs=total_jobs,
            total_bids=total_bids,
            bid_efficiency=total_filled / total_bids if total_bids > 0 else 0,
            avg_bids_per_job=total_bids / total_jobs if total_jobs > 0 else 0,
            participation_rate=np.mean(participation_rates) if participation_rates else 0,
            rejection_rate=np.mean(rejection_rates) if rejection_rates else 0,
            freelancer_hiring_rate=(
                successful_freelancers / total_freelancers 
                if total_freelancers > 0 else 0
            ),
            gini_coefficient=gini,
            market_health_score=market_health
        )
    
    @staticmethod
    def calculate_gini_coefficient(values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values or all(v == 0 for v in values):
            return 0.0
            
        values = np.array(sorted(values))
        n = len(values)
        cumsum = np.cumsum(values)
        
        if cumsum[-1] == 0:  # All values are zero
            return 0.0
            
        return (
            (2 * np.sum((np.arange(1, n + 1) * values))) / (n * cumsum[-1]) - (n + 1) / n
        )


class TimeSeriesExtractor:
    """Utility class for extracting time series data from simulations"""
    
    @staticmethod
    def extract_round_metrics(simulation_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract round-by-round metrics as DataFrame"""
        round_data = simulation_data.get('round_data', [])
        
        rounds_df = pd.DataFrame([{
            'round': r.get('round', idx + 1),
            'jobs_posted': r.get('jobs_posted', 0),
            'total_bids': r.get('total_bids', 0),
            'jobs_filled_this_round': r.get('jobs_filled_this_round', 0),
            'jobs_filled_cumulative': r.get('jobs_filled', 0),
            'bid_rejection_rate': r.get('bid_rejection_metrics', {}).get('bid_rejection_rate', 0),
            'participation_rate': r.get('market_activity', {}).get('freelancer_participation_rate', 0),
            'avg_bids_per_job': r.get('bid_distribution', {}).get('avg_bids_per_job', 0),
            'supply_demand_ratio': r.get('market_health', {}).get('supply_demand_ratio', 0),
            'health_score': r.get('market_health', {}).get('health_score', 0),
        } for idx, r in enumerate(round_data)])
        
        return rounds_df


class StatisticalAnalysis:
    """Utility class for statistical calculations"""
    
    @staticmethod
    def calculate_confidence_interval(
        values: List[float], 
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """Calculate mean and confidence interval"""
        if len(values) <= 1:
            mean_val = np.mean(values) if values else 0
            return mean_val, mean_val, mean_val
        
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        # 95% confidence interval using t-distribution
        t_val = stats.t.ppf((1 + confidence_level) / 2, len(values) - 1)
        margin = t_val * std_val / np.sqrt(len(values))
        
        ci_lower = mean_val - margin
        ci_upper = mean_val + margin
        
        return mean_val, ci_lower, ci_upper
    
    @staticmethod
    def perform_t_test(values1: List[float], values2: List[float]) -> Tuple[float, float]:
        """Perform independent t-test between two groups"""
        if len(values1) < 2 or len(values2) < 2:
            return 0.0, 1.0
        
        t_stat, p_value = stats.ttest_ind(values1, values2)
        return t_stat, p_value


class ConfigurationParser:
    """Utility class for parsing simulation configurations"""
    
    @staticmethod
    def extract_config_key(file_path: str) -> str:
        """Extract configuration key from simulation file"""
        try:
            # Simple line-by-line parsing to extract agent types and key parameters
            freelancer_type = 'unknown'
            client_type = 'unknown'
            bid_probability = 'unknown'
            reflections_enabled = 'unknown'
            cooldown_min = 'unknown'
            cooldown_max = 'unknown'
            relevance_mode = 'unknown'
            job_selection_method = 'unknown'
            
            with open(file_path, 'r') as f:
                for line in f:
                    if '"freelancer_agent_type"' in line:
                        freelancer_type = line.split(':')[1].strip().strip(',').strip('"')
                    elif '"client_agent_type"' in line:
                        client_type = line.split(':')[1].strip().strip(',').strip('"')
                    elif '"random_freelancer_bid_probability"' in line:
                        bid_probability = line.split(':')[1].strip().strip(',').strip()
                    elif '"enable_reflections"' in line:
                        reflections_enabled = line.split(':')[1].strip().strip(',').strip()
                    elif '"job_posting_cooldown_min"' in line:
                        cooldown_min = line.split(':')[1].strip().strip(',').strip()
                    elif '"job_posting_cooldown_max"' in line:
                        cooldown_max = line.split(':')[1].strip().strip(',').strip()
                    elif '"relevance_mode"' in line:
                        relevance_mode = line.split(':')[1].strip().strip(',').strip('"')
                    elif '"job_selection_method"' in line:
                        job_selection_method = line.split(':')[1].strip().strip(',').strip('"')
                    elif '"freelancer_profiles"' in line:
                        # We've reached the end of config, stop reading
                        break
            
            # Create comprehensive configuration key
            config_key = (
                f"{freelancer_type}_{client_type}_bid{bid_probability}_"
                f"refl{reflections_enabled}_cool{cooldown_min}-{cooldown_max}_"
                f"rel{relevance_mode}_job{job_selection_method}"
            )
            
            return config_key
            
        except Exception as e:
            return 'unknown_unknown_bidunknown_reflunknown_coolunknown-unknown_relunknown_jobunknown'
    
    @staticmethod
    def find_varying_parameters(config_keys: List[str]) -> set:
        """Identify which parameters vary across configurations"""
        if not config_keys:
            return set()
        
        # Extract all parameter values for each configuration
        all_params = []
        for config_key in config_keys:
            parts = config_key.split('_')
            if len(parts) >= 2:
                params = {
                    'freelancer_type': parts[0],
                    'client_type': parts[1],
                    'bid_prob': None,
                    'refl_val': None,
                    'cooldown_val': None,
                    'relevance_val': None,
                    'job_method': None
                }
                
                for part in parts[2:]:
                    if part.startswith('bid'):
                        params['bid_prob'] = part[3:]
                    elif part.startswith('refl'):
                        params['refl_val'] = part[4:]
                    elif part.startswith('cool'):
                        params['cooldown_val'] = part[4:]
                    elif part.startswith('rel'):
                        params['relevance_val'] = part[3:]
                    elif part.startswith('job'):
                        params['job_method'] = part[3:]
                
                all_params.append(params)
        
        # Find parameters that vary
        varying_params = set()
        if len(all_params) > 1:
            first_config = all_params[0]
            for param_name in first_config.keys():
                values = [config.get(param_name) for config in all_params]
                if len(set(values)) > 1:  # Parameter has different values
                    varying_params.add(param_name)
        
        return varying_params
    
    @staticmethod
    def get_readable_config_name(config_key: str, varying_params: set = None) -> str:
        """Convert config key to readable format, showing only varying parameters"""
        parts = config_key.split('_')
        
        if len(parts) >= 2:
            freelancer_type = parts[0].title()
            client_type = parts[1].title()
            
            # Always show agent types as they're the primary distinguisher
            readable_name = f"{freelancer_type}-{client_type}"
            
            # If no varying_params specified, show all parameters (backward compatibility)
            if varying_params is None:
                varying_params = {'freelancer_type', 'client_type', 'bid_prob', 'refl_val', 
                                'cooldown_val', 'relevance_val', 'job_method'}
            
            # Extract and show only varying parameters
            for part in parts[2:]:
                if part.startswith('bid') and 'bid_prob' in varying_params:
                    bid_prob = part[3:]  # Remove 'bid' prefix
                    try:
                        bid_val = float(bid_prob)
                        readable_name += f" (bid:{bid_val:.1f})"
                    except ValueError:
                        if bid_prob != 'unknown':
                            readable_name += f" (bid:{bid_prob})"
                elif part.startswith('refl') and 'refl_val' in varying_params:
                    refl_val = part[4:]  # Remove 'refl' prefix
                    if refl_val.lower() == 'true':
                        readable_name += " +Refl"
                    elif refl_val.lower() == 'false':
                        readable_name += " -Refl"
                elif part.startswith('cool') and 'cooldown_val' in varying_params:
                    cooldown_val = part[4:]  # Remove 'cool' prefix
                    if cooldown_val != 'unknown-unknown' and 'unknown' not in cooldown_val:
                        readable_name += f" [cool:{cooldown_val}]"
                elif part.startswith('rel') and 'relevance_val' in varying_params:
                    relevance_val = part[3:]  # Remove 'rel' prefix
                    if relevance_val != 'unknown':
                        readable_name += f" ({relevance_val})"
                elif part.startswith('job') and 'job_method' in varying_params:
                    job_method = part[3:]  # Remove 'job' prefix
                    if job_method != 'unknown':
                        readable_name += f" <{job_method}>"
            
            return readable_name
        
        # Fallback for old format
        mapping = {
            'random_random': 'Random-Random',
            'random_llm': 'Random-LLM', 
            'llm_random': 'LLM-Random',
            'llm_llm': 'LLM-LLM'
        }
        return mapping.get(config_key, config_key)


class DataLoader:
    """Utility class for loading simulation data"""
    
    @staticmethod
    def load_simulation_data(file_path: str) -> Dict[str, Any]:
        """Load simulation data from file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def discover_simulation_files(results_dir: str) -> List[str]:
        """Discover all simulation files in results directory"""
        results_path = Path(results_dir)
        return [str(f) for f in results_path.glob("*.json")]


# Color palettes for consistent visualization (integrating with existing style)
CONFIG_COLORS = {
    'llm_llm': '#2E86AB',      # Blue
    'random_random': '#A23B72', # Magenta  
    'random_llm': '#F18F01',    # Orange
    'llm_random': '#C73E1D',    # Red
    'greedy_greedy': '#4CAF50', # Green
    'greedy_llm': '#9C27B0',    # Purple
}

METRIC_COLORS = {
    'fill_rate': '#1f77b4',
    'bid_efficiency': '#ff7f0e',
    'participation_rate': '#2ca02c',
    'rejection_rate': '#d62728',
    'market_health': '#9467bd'
}
