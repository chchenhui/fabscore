"""
Data Analysis Module for Marketplace Simulation
Generates scientific insights and visualizations for the research paper
"""

import pandas as pd
from scipy import stats
import json
import numpy as np
import logging
from datetime import timedelta
from typing import Dict
from pathlib import Path
from ...marketplace.job_categories import JobCategory, category_manager

logger = logging.getLogger(__name__)

class MarketplaceAnalyzer:
    """Analyzes simulation results for scientific insights"""
    
    def __init__(self, log_file: str):
        """Initialize analyzer with a specific simulation log file"""
        self.log_file = Path(log_file)
        self.round_data = None
        self.bid_data = None
        self.job_data = None
        self.freelancer_states = None
        self.hiring_outcomes = None
        
    def _get_hired_job_ids(self):
        """Helper method to extract job IDs that were actually hired"""
        hired_job_ids = set()
        if hasattr(self, 'hiring_outcomes') and self.hiring_outcomes is not None:
            hiring_data = self.hiring_outcomes
            
            # Handle both DataFrame and list formats
            if isinstance(hiring_data, pd.DataFrame):
                if len(hiring_data) > 0:
                    successful_hires = hiring_data[
                        (hiring_data['selected_freelancer'].notna()) & 
                        (hiring_data['selected_freelancer'] != 'none')
                    ]
                    hired_job_ids = set(successful_hires['job_id'].unique())
            elif isinstance(hiring_data, list):
                for outcome in hiring_data:
                    if (outcome.get('selected_freelancer') and 
                        outcome['selected_freelancer'] != 'none'):
                        hired_job_ids.add(outcome['job_id'])
        
        return hired_job_ids
        
    def load_data(self):
        """Load simulation data from the log file"""
        if not self.log_file.exists():
            logger.error(f"Log file not found: {self.log_file}")
            return
            
        with open(self.log_file) as f:
            simulation_data = json.load(f)
            
            # Load freelancer data
            freelancer_profiles = simulation_data.get('freelancer_profiles', {})
            self.freelancer_states = pd.DataFrame.from_dict(freelancer_profiles, orient='index')
            
            # Load round data
            self.round_data = pd.DataFrame(simulation_data.get('round_data', []))
            
            # Load job data and normalize categories
            jobs = simulation_data.get('all_jobs', [])
            for job in jobs:
                if isinstance(job.get('category'), list):
                    job['category'] = job['category'][0]  # Take first category if it's a list
            self.job_data = pd.DataFrame(jobs)
            
            # Load hiring outcomes to determine which jobs were actually filled
            hiring_outcomes = simulation_data.get('hiring_outcomes', [])
            self.hiring_outcomes = pd.DataFrame(hiring_outcomes)
            
            # Load bid data
            bids = simulation_data.get('all_bids', [])
            self.bid_data = pd.DataFrame(bids)
            
            # Load hiring outcomes (handle both new dict format and legacy string format)
            raw_hiring_outcomes = simulation_data.get('hiring_outcomes', [])
            self.hiring_outcomes = []
            
            for outcome in raw_hiring_outcomes:
                if isinstance(outcome, str):
                    # Parse legacy string representation of HiringDecision
                    if outcome.startswith('HiringDecision('):
                        # Extract values from string representation
                        import re
                        job_id_match = re.search(r"job_id='([^']*)'", outcome)
                        client_id_match = re.search(r"client_id='([^']*)'", outcome)
                        freelancer_match = re.search(r"selected_freelancer=([^,)]*)", outcome)
                        reasoning_match = re.search(r"reasoning='([^']*)'", outcome)
                        
                        job_id = job_id_match.group(1) if job_id_match else None
                        client_id = client_id_match.group(1) if client_id_match else None
                        selected_freelancer = freelancer_match.group(1) if freelancer_match else None
                        if selected_freelancer and selected_freelancer.strip("'") == "None":
                            selected_freelancer = None
                        elif selected_freelancer and selected_freelancer.startswith("'"):
                            selected_freelancer = selected_freelancer.strip("'")
                        reasoning = reasoning_match.group(1) if reasoning_match else ""
                        
                        parsed_outcome = {
                            'job_id': job_id,
                            'client_id': client_id,
                            'selected_freelancer': selected_freelancer,
                            'reasoning': reasoning
                        }
                        self.hiring_outcomes.append(parsed_outcome)
                else:
                    # Handle new dict format
                    self.hiring_outcomes.append(outcome)
            
            # Add freelancer skills to bid data for analysis
            if not self.bid_data.empty:
                freelancer_skills = {
                    fid: profile.get('skills', []) 
                    for fid, profile in freelancer_profiles.items()
                }
                self.bid_data['freelancer_skills'] = self.bid_data['freelancer_id'].map(freelancer_skills)
                
                # Add winner information to bids
                winner_bids = set()
                for outcome in self.hiring_outcomes:
                    if outcome.get('selected_freelancer'):
                        # Find the winning bid
                        job_id = outcome['job_id']
                        freelancer_id = outcome['selected_freelancer']
                        winning_bid = self.bid_data[
                            (self.bid_data['job_id'] == job_id) & 
                            (self.bid_data['freelancer_id'] == freelancer_id)
                        ]
                        if not winning_bid.empty:
                            winner_bids.update(winning_bid.index)
                
                self.bid_data['is_winner'] = self.bid_data.index.isin(winner_bids)
        
        logger.info(f"Loaded data: {len(self.round_data)} rounds, {len(self.bid_data)} bids, {len(self.job_data)} jobs")
    
    def generate_market_efficiency_analysis(self):
        """Analyze market efficiency over time"""
        if self.round_data is None:
            return None
        
        # Calculate efficiency metrics
        if self.bid_data is not None and self.job_data is not None and not self.bid_data.empty:
            # Calculate bids per job for each round
            job_bids = self.bid_data.groupby('job_id').size()
            self.job_data['num_bids'] = self.job_data['id'].map(job_bids).fillna(0)
        elif self.job_data is not None:
            # No bids data, all jobs have 0 bids
            self.job_data['num_bids'] = 0
            
        # Calculate if job was filled (actually hired someone)
        if self.job_data is not None:
            hired_job_ids = self._get_hired_job_ids()
            if hired_job_ids:
                self.job_data['filled'] = self.job_data['id'].isin(hired_job_ids)
            else:
                # Fallback to old logic if no hiring outcomes available
                self.job_data['filled'] = self.job_data['num_bids'] > 0
            
            # Group by round
            # Handle empty groups to avoid numpy warnings
            grouped = self.job_data.groupby('posted_time')
            round_metrics = pd.DataFrame()
            
            # Initialize metrics with default values
            metrics = {}
            
            # Calculate metrics with explicit error handling
            for name, group in grouped:
                metrics[name] = {
                    'filled': 0.0,
                    'num_bids_count': 0,
                    'num_bids_mean': 0.0,
                    'num_bids_sum': 0.0
                }
                
                try:
                    if len(group) > 0:
                        # Calculate filled rate
                        try:
                            metrics[name]['filled'] = float(group['filled'].mean())
                        except (TypeError, ValueError):
                            pass  # Keep default value
                        
                        # Calculate bid count
                        metrics[name]['num_bids_count'] = len(group)
                        
                        # Calculate bid mean
                        try:
                            metrics[name]['num_bids_mean'] = float(group['num_bids'].mean())
                        except (TypeError, ValueError):
                            pass  # Keep default value
                        
                        # Calculate bid sum
                        try:
                            metrics[name]['num_bids_sum'] = float(group['num_bids'].sum())
                        except (TypeError, ValueError):
                            pass  # Keep default value
                except Exception:
                    pass  # Keep default values
            
            # Convert to DataFrame with MultiIndex columns
            round_metrics = pd.DataFrame.from_dict(metrics, orient='index')
            
            # Ensure numeric values
            for col in round_metrics.columns:
                round_metrics[col] = pd.to_numeric(round_metrics[col], errors='coerce').fillna(0.0)
            
            # Create MultiIndex columns
            round_metrics.columns = pd.MultiIndex.from_tuples([
                ('filled', ''),
                ('num_bids', 'count'),
                ('num_bids', 'mean'),
                ('num_bids', 'sum')
            ])
            
            # Handle empty slices to avoid numpy warnings
            efficiency_metrics = {
                'round': list(range(len(round_metrics))),
                'job_fill_rate': round_metrics['filled'].fillna(0.0).values.astype(float).tolist(),
                'avg_bids_per_job': round_metrics[('num_bids', 'mean')].fillna(0.0).values.astype(float).tolist(),
                'market_activity': round_metrics[('num_bids', 'sum')].fillna(0.0).values.astype(float).tolist()
            }
        else:
            efficiency_metrics = {
                'round': [],
                'job_fill_rate': [],
                'avg_bids_per_job': [],
                'market_activity': []
            }
        
        # Convert to DataFrame and ensure numeric types
        df = pd.DataFrame(efficiency_metrics)
        for col in ['job_fill_rate', 'avg_bids_per_job', 'market_activity']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def analyze_skill_distribution(self):
        """Analyze category distribution and performance across freelancer pool"""
        if self.freelancer_states is None:
            return None
            
        # Initialize analysis structure
        category_analysis = {
            'category_coverage': {},  # Count and performance by category
            'category_competition': {},  # Average bids per job by category
            'category_rates': {},  # Average rates by category
            'category_success': {},  # Success rates by category
        }
        
        # Get all unique categories
        job_categories = set(self.job_data['category'].unique())
        # Get categories where freelancers have expertise based on their assigned category
        freelancer_categories = set(self.freelancer_states['category'].unique())
        all_categories = job_categories.union(freelancer_categories)
        
        # Analyze each category
        for category in all_categories:
            # Get jobs in this category
            category_jobs = self.job_data[self.job_data['category'] == category]
            
            # Get freelancers with significant expertise in this category
            # Find the JobCategory enum by title
            job_category = next((cat for cat in JobCategory if cat.title == category), None)
            if job_category:
                category_freelancers = self.freelancer_states[
                    self.freelancer_states['category'] == category
                ]
            else:
                # If category not found, assume no freelancers match
                category_freelancers = self.freelancer_states[self.freelancer_states['id'].isna()]
            
            metrics = {
                'total_jobs': len(category_jobs),
                'total_freelancers': len(category_freelancers),
                'avg_bids_per_job': 0,
                'fill_rate': 0,
                'avg_job_rate': 0,
                'success_rate': 0
            }
            
            if len(category_jobs) > 0:
                # Calculate job metrics
                if self.bid_data is not None and not self.bid_data.empty:
                    category_bids = self.bid_data[self.bid_data['job_id'].isin(category_jobs['id'])]
                else:
                    category_bids = []
                # Handle empty slices to avoid numpy warnings
                metrics['avg_bids_per_job'] = len(category_bids) / len(category_jobs)
                metrics['fill_rate'] = float(category_jobs['filled'].mean()) if len(category_jobs) > 0 else 0.0
                metrics['avg_job_rate'] = float(category_jobs['budget_amount'].mean()) if len(category_jobs) > 0 else 0.0
                
                # Calculate success rate (completed jobs / total jobs)
                if len(category_freelancers) > 0:
                    # Handle empty slices to avoid numpy warnings
                    total_completed = float(category_freelancers['completed_jobs'].sum()) if len(category_freelancers) > 0 else 0.0
                    metrics['success_rate'] = total_completed / len(category_jobs) if len(category_jobs) > 0 else 0.0
            
            category_analysis['category_coverage'][category] = metrics
        
        # Calculate competition by category
        if self.job_data is not None and 'num_bids' in self.job_data.columns:
            # Handle empty groups to avoid numpy warnings
            grouped = self.job_data.groupby('category')
            category_bids = {}
            for category in grouped.groups:
                group = grouped.get_group(category)
                category_bids[category] = float(group['num_bids'].mean()) if len(group) > 0 else 0.0
            category_analysis['category_competition'] = category_bids
        
        # Calculate average rates by category
        if self.bid_data is not None:
            category_rates = {}
            for category in all_categories:
                category_jobs = self.job_data[self.job_data['category'] == category]
                if len(category_jobs) > 0:
                    if self.bid_data is not None and not self.bid_data.empty:
                        category_bids = self.bid_data[self.bid_data['job_id'].isin(category_jobs['id'])]
                    else:
                        category_bids = []
                    if len(category_bids) > 0:
                        # Handle empty slices to avoid numpy warnings
                        category_rates[category] = {
                            'avg_bid_rate': float(category_bids['proposed_rate'].mean()) if len(category_bids) > 0 else 0.0,
                            'min_bid_rate': float(category_bids['proposed_rate'].min()) if len(category_bids) > 0 else 0.0,
                            'max_bid_rate': float(category_bids['proposed_rate'].max()) if len(category_bids) > 0 else 0.0
                        }
            category_analysis['category_rates'] = category_rates
        
        # Calculate success rates by category
        if self.freelancer_states is not None:
            category_success = {}
            for category in all_categories:
                # Get freelancers with significant expertise in this category
                # Find the JobCategory enum by title
                job_category = next((cat for cat in JobCategory if cat.title == category), None)
                if job_category:
                    # Simple approach: just include all freelancers for now (basic analysis)
                    category_freelancers = self.freelancer_states
                else:
                    # If category not found, assume no freelancers match
                    category_freelancers = self.freelancer_states[self.freelancer_states['id'].isna()]
                if len(category_freelancers) > 0:
                    # Handle empty slices to avoid numpy warnings
                    category_success[category] = {
                        'avg_completed_jobs': float(category_freelancers['completed_jobs'].mean()) if len(category_freelancers) > 0 else 0.0,
                        'total_completed_jobs': float(category_freelancers['completed_jobs'].sum()) if len(category_freelancers) > 0 else 0.0,
                        'avg_success_rate': float((category_freelancers['completed_jobs'] / category_freelancers['total_bids']).mean()) if len(category_freelancers) > 0 else 0.0
                    }
            category_analysis['category_success'] = category_success
        
        # Calculate category specialization metrics
        if self.bid_data is not None:
            category_specialization = {}
            category_cross_bidding = {}
            
            for category in all_categories:
                # Get freelancers specializing in this category
                # Get freelancers with high expertise in this category
                # Find the JobCategory enum by title
                job_category = next((cat for cat in JobCategory if cat.title == category), None)
                if job_category:
                    specialists = self.freelancer_states[
                        self.freelancer_states['category'] == category
                    ]
                else:
                    # If category not found, assume no specialists
                    specialists = self.freelancer_states[self.freelancer_states['id'].isna()]
                
                if len(specialists) > 0:
                    # Get bids by specialists
                    if self.bid_data is not None and not self.bid_data.empty:
                        specialist_bids = self.bid_data[self.bid_data['freelancer_id'].isin(specialists.index)]
                    else:
                        specialist_bids = pd.DataFrame()
                    
                    # Calculate how often specialists bid in their category vs others
                    # Create a copy to avoid SettingWithCopyWarning
                    if not specialist_bids.empty:
                        specialist_bids = specialist_bids.copy()
                        specialist_bids.loc[:, 'bid_category'] = specialist_bids['job_id'].map(self.job_data.set_index('id')['category'])
                        in_category_bids = specialist_bids[specialist_bids['bid_category'] == category]
                        out_category_bids = specialist_bids[specialist_bids['bid_category'] != category]
                    else:
                        in_category_bids = pd.DataFrame()
                        out_category_bids = pd.DataFrame()
                    
                    specialization_metrics = {
                        'in_category_bid_rate': len(in_category_bids) / len(specialist_bids) if len(specialist_bids) > 0 else 0,
                        'out_category_bid_rate': len(out_category_bids) / len(specialist_bids) if len(specialist_bids) > 0 else 0,
                        # Handle empty slices to avoid numpy warnings
                        'in_category_success_rate': float(in_category_bids.get('is_winner', pd.Series()).mean()) if len(in_category_bids) > 0 else 0.0,
                        'out_category_success_rate': float(out_category_bids.get('is_winner', pd.Series()).mean()) if len(out_category_bids) > 0 else 0.0
                    }
                    category_specialization[category] = specialization_metrics
                    
                    # Analyze which other categories specialists bid in
                    if len(out_category_bids) > 0:
                        cross_bid_distribution = out_category_bids['bid_category'].value_counts().to_dict()
                        category_cross_bidding[category] = cross_bid_distribution
            
            category_analysis['category_specialization'] = category_specialization
            category_analysis['category_cross_bidding'] = category_cross_bidding
        
        return category_analysis
    
    def analyze_persona_performance(self):
        """Compare performance across different persona types"""
        if self.freelancer_states is None or len(self.freelancer_states) == 0:
            return None
            
        # Calculate success metrics
        metrics = {
            'total_bids': float(self.freelancer_states['total_bids'].sum()),
            'total_completed': float(self.freelancer_states['completed_jobs'].sum()),
            'avg_bids_per_freelancer': float(self.freelancer_states['total_bids'].mean()),
            'avg_completed_per_freelancer': float(self.freelancer_states['completed_jobs'].mean()),
            'success_rate': float(self.freelancer_states['completed_jobs'].sum()) / float(self.freelancer_states['total_bids'].sum()) if float(self.freelancer_states['total_bids'].sum()) > 0 else 0.0
        }
        
        return metrics
    
    def analyze_job_complexity(self):
        """Analyze the relationship between job complexity and fill rates"""
        if self.job_data is None:
            return None
            
        complexity_analysis = {
            'category_complexity': {},  # Complexity metrics by category
            'budget_impact': {},  # How budget level affects fill rate
            'skill_impact': {},  # How required skills affect fill rate
            'complexity_patterns': {}  # Overall patterns in job complexity
        }
        
        # Calculate job complexity based on category and budget
        def calculate_job_complexity(row):
            category = row['category']
            budget = row['budget_amount']
            skills = row['skills_required']
            
            # Get category average budget
            category_jobs = self.job_data[self.job_data['category'] == category]
            
            # Handle empty slices to avoid numpy warnings
            try:
                category_avg_budget = float(category_jobs['budget_amount'].mean()) if len(category_jobs) > 0 else 0.0
            except (TypeError, ValueError):
                category_avg_budget = 0.0
            
            # Complexity factors:
            # 1. Budget factor (normalized to 0-1)
            # Higher budget relative to category average indicates more complexity
            if category_avg_budget > 0:
                budget_factor = min(float(budget) / category_avg_budget, 2) / 2  # Cap at 2x average
            else:
                budget_factor = 0.5  # Default to middle if no reference
            
            # 2. Skills count factor (normalized to 0-1)
            # Compare to typical skill count for this category
            try:
                category_avg_skills = float(category_jobs['skills_required'].apply(len).mean()) if len(category_jobs) > 0 else 0.0
            except (TypeError, ValueError):
                category_avg_skills = 0.0
                
            if category_avg_skills > 0:
                skill_factor = min(len(skills) / category_avg_skills, 2) / 2  # Cap at 2x average
            else:
                skill_factor = len(skills) / 4  # Fallback to assuming max 4 skills
            
            # 3. Competition factor (normalized to 0-1)
            # More bids indicate a more competitive/complex job
            if self.bid_data is not None and not self.bid_data.empty:
                job_bids = len(self.bid_data[self.bid_data['job_id'] == row['id']])
            else:
                job_bids = 0
            
            try:
                if self.bid_data is not None and not self.bid_data.empty:
                    category_bids = len(self.bid_data[self.bid_data['job_id'].isin(category_jobs['id'])])
                else:
                    category_bids = 0
                category_avg_bids = float(category_bids) / len(category_jobs) if len(category_jobs) > 0 else 0.0
            except (TypeError, ValueError):
                category_avg_bids = 0.0
                
            if category_avg_bids > 0:
                competition_factor = min(float(job_bids) / category_avg_bids, 2) / 2  # Cap at 2x average
            else:
                competition_factor = 0.5
            
            # Combined complexity score (0-1)
            weights = {
                'budget': 0.4,  # Budget is a strong indicator
                'skills': 0.3,  # Number of required skills
                'competition': 0.3  # Market interest level
            }
            
            complexity = (
                weights['budget'] * budget_factor +
                weights['skills'] * skill_factor +
                weights['competition'] * competition_factor
            )
            
            return complexity, budget_factor, skill_factor, competition_factor
            
        # Calculate complexity scores for each job
        complexity_scores = []
        for _, job in self.job_data.iterrows():
            score, budget_f, skill_f, comp_f = calculate_job_complexity(job)
            complexity_scores.append({
                'job_id': job['id'],
                'category': job['category'],
                'complexity_score': score,
                'budget_factor': budget_f,
                'skill_factor': skill_f,
                'competition_factor': comp_f
            })
        
        complexity_df = pd.DataFrame(complexity_scores)
        self.job_data['complexity_score'] = complexity_df.set_index('job_id')['complexity_score']
        
        # Analyze complexity by category
        for category in self.job_data['category'].unique():
            category_jobs = self.job_data[self.job_data['category'] == category]
            category_scores = complexity_df[complexity_df['category'] == category]
            
            if len(category_jobs) > 0:
                # Handle empty arrays to avoid numpy warnings
                metrics = {
                    'avg_complexity': 0.0,
                    'avg_budget_factor': 0.0,
                    'avg_skill_factor': 0.0,
                    'avg_competition_factor': 0.0,
                    'fill_rate_by_complexity': {}
                }
                
                # Calculate averages with explicit error handling
                for metric in ['complexity_score', 'budget_factor', 'skill_factor', 'competition_factor']:
                    try:
                        if len(category_scores) > 0:
                            avg_val = float(category_scores[metric].mean())
                            metrics[f'avg_{metric}' if metric != 'complexity_score' else 'avg_complexity'] = avg_val
                    except (TypeError, ValueError):
                        pass  # Keep default 0.0
                
                # Calculate fill rates for different complexity levels
                try:
                    category_jobs['complexity_level'] = pd.qcut(
                        category_jobs['complexity_score'],
                        q=3,
                        labels=['Low', 'Medium', 'High']
                    )
                    # Handle empty groups to avoid numpy warnings
                    grouped = category_jobs.groupby('complexity_level')
                    fill_rates = {}
                    for level in ['Low', 'Medium', 'High']:
                        if level in grouped.groups:
                            group = grouped.get_group(level)
                            fill_rates[level] = float(group['filled'].mean()) if len(group) > 0 else 0.0
                        else:
                            fill_rates[level] = 0.0
                    metrics['fill_rate_by_complexity'] = fill_rates
                except ValueError:
                    # If qcut fails (e.g., too few unique values), use simpler approach
                    median = category_jobs['complexity_score'].median()
                    # Create a copy to avoid SettingWithCopyWarning
                    category_jobs = category_jobs.copy()
                    category_jobs.loc[:, 'complexity_level'] = category_jobs['complexity_score'].apply(
                        lambda x: 'High' if x > median else 'Low'
                    )
                    # Handle empty groups to avoid numpy warnings
                    grouped = category_jobs.groupby('complexity_level')
                    fill_rates = {}
                    for level in ['Low', 'High']:
                        if level in grouped.groups:
                            group = grouped.get_group(level)
                            fill_rates[level] = float(group['filled'].mean()) if len(group) > 0 else 0.0
                        else:
                            fill_rates[level] = 0.0
                    metrics['fill_rate_by_complexity'] = fill_rates
                
                complexity_analysis['category_complexity'][category] = metrics
        
        # Analyze budget impact by category
        budget_impact = {}
        for category in self.job_data['category'].unique():
            category_jobs = self.job_data[self.job_data['category'] == category]
            if len(category_jobs) > 0:
                # Calculate relative budget levels
                category_avg = category_jobs['budget_amount'].mean()
                # Create a copy to avoid SettingWithCopyWarning
                category_jobs = category_jobs.copy()
                category_jobs.loc[:, 'budget_level'] = category_jobs['budget_amount'].apply(
                    lambda x: 'Above Average' if x > category_avg else 'Below Average'
                )
                
                # Handle empty groups to avoid numpy warnings
                grouped = category_jobs.groupby('budget_level')
                budget_metrics = pd.DataFrame()
                for metric in ['filled', 'num_bids', 'complexity_score']:
                    budget_metrics[metric] = grouped[metric].agg(lambda x: float(x.mean()) if len(x) > 0 else 0.0)
                
                budget_impact[category] = budget_metrics.to_dict()
        
        complexity_analysis['budget_impact'] = budget_impact
        
        # Analyze skill impact by category
        skill_impact = {}
        for category in self.job_data['category'].unique():
            category_jobs = self.job_data[self.job_data['category'] == category]
            if len(category_jobs) > 0:
                avg_skills = category_jobs['skills_required'].apply(len).mean()
                # Create a copy to avoid SettingWithCopyWarning
                category_jobs = category_jobs.copy()
                category_jobs.loc[:, 'skill_level'] = category_jobs['skills_required'].apply(
                    lambda x: 'Above Average' if len(x) > avg_skills else 'Below Average'
                )
                
                # Handle empty groups to avoid numpy warnings
                grouped = category_jobs.groupby('skill_level')
                skill_metrics = pd.DataFrame()
                for metric in ['filled', 'num_bids', 'complexity_score']:
                    skill_metrics[metric] = grouped[metric].agg(lambda x: float(x.mean()) if len(x) > 0 else 0.0)
                
                skill_impact[category] = skill_metrics.to_dict()
        
        complexity_analysis['skill_impact'] = skill_impact
        
        # Calculate overall complexity patterns
        patterns = {}
        for category in self.job_data['category'].unique():
            category_jobs = self.job_data[self.job_data['category'] == category]
            if len(category_jobs) > 0:
                # Calculate correlations
                def safe_correlation(x, y):
                    if len(x) < 3 or x.std() == 0 or y.std() == 0:
                        return {'correlation': None, 'p_value': None, 'significance': 'insufficient_data'}
                    
                    correlation, p_value = stats.pearsonr(x, y)
                    
                    # Determine significance level
                    if p_value < 0.01:
                        significance = 'strong'
                    elif p_value < 0.05:
                        significance = 'moderate'
                    elif p_value < 0.1:
                        significance = 'weak'
                    else:
                        significance = 'not_significant'
                    
                    return {
                        'correlation': float(correlation),
                        'p_value': float(p_value),
                        'significance': significance
                    }
                
                patterns[category] = {
                    'complexity_fill_rate': safe_correlation(
                        category_jobs['complexity_score'],
                        category_jobs['filled']
                    ),
                    'complexity_bids': safe_correlation(
                        category_jobs['complexity_score'],
                        category_jobs['num_bids']
                    ),
                    'budget_bids': safe_correlation(
                        category_jobs['budget_amount'],
                        category_jobs['num_bids']
                    ),
                    'sample_size': len(category_jobs)
                }
        
        complexity_analysis['complexity_patterns'] = patterns
        
        return complexity_analysis
    
    def analyze_bidding_strategies(self):
        """Analyze bidding patterns and success by category"""
        if self.bid_data is None or self.job_data is None or self.bid_data.empty:
            return pd.DataFrame()  # Return empty DataFrame instead of None for consistent structure
        
        # Add category information to bids
        self.bid_data['category'] = self.bid_data['job_id'].map(self.job_data.set_index('id')['category'])
        
        # Calculate success metrics for each bid
        self.bid_data['is_winner'] = False
        for job_id in self.bid_data['job_id'].unique():
            job_bids = self.bid_data[self.bid_data['job_id'] == job_id]
            if len(job_bids) > 0:
                # Winner is the highest bid (in a real system, this would be the selected bid)
                winner_id = job_bids.sort_values('proposed_rate', ascending=False).iloc[0].name
                self.bid_data.loc[winner_id, 'is_winner'] = True
        
        # Analyze bidding patterns by category
        category_metrics = []
        for category in self.bid_data['category'].unique():
            # Get all bids in this category
            category_bids = self.bid_data[self.bid_data['category'] == category].copy()
            
            # Handle empty slices to avoid numpy warnings
            metrics = {
                'category': category,
                'total_bids': len(category_bids)
            }
            
            # Calculate rate metrics
            if len(category_bids) > 0:
                metrics.update({
                    'avg_bid_rate': float(category_bids['proposed_rate'].mean()) if len(category_bids) > 0 else 0.0
                })
            
            # Calculate success metrics
            # Handle empty slices to avoid numpy warnings
            if len(category_bids) > 0:
                total_wins = float(category_bids['is_winner'].sum())
                win_rate = float(category_bids['is_winner'].mean())
            else:
                total_wins = 0.0
                win_rate = 0.0
            metrics.update({
                'total_wins': total_wins,
                'win_rate': win_rate
            })
            
            # Calculate bid timing and position metrics
            if len(category_bids) > 0:
                # Calculate positions for all bids
                category_bids.loc[:, 'bid_position'] = category_bids.groupby('job_id').cumcount() + 1
                
                # Handle empty slices to avoid numpy warnings
                bid_positions = category_bids.groupby('job_id').cumcount()
                metrics.update({
                    'avg_bid_position': float(bid_positions.mean()) + 1 if len(bid_positions) > 0 else 1.0
                })

            category_metrics.append(metrics)
        
        return pd.DataFrame(category_metrics)
    
    def analyze_unfilled_jobs(self):
        """Analyze characteristics of unfilled jobs to understand low fill rates"""
        if self.job_data is None:
            return None
            
        # Calculate fill status if not already done
        if 'num_bids' not in self.job_data.columns:
            if self.bid_data is not None and not self.bid_data.empty:
                job_bid_counts = self.bid_data.groupby('job_id').size()
                self.job_data['num_bids'] = self.job_data['id'].map(job_bid_counts).fillna(0)
            else:
                self.job_data['num_bids'] = 0
        if 'filled' not in self.job_data.columns:
            # Calculate if job was filled (actually hired someone)
            hired_job_ids = self._get_hired_job_ids()
            if hired_job_ids:
                self.job_data['filled'] = self.job_data['id'].isin(hired_job_ids)
            else:
                # Fallback to old logic if no hiring outcomes available
                self.job_data['filled'] = self.job_data['num_bids'] > 0
            
        unfilled_analysis = {
            # Handle empty slices to avoid numpy warnings
            'overall_stats': {
                'total_jobs': len(self.job_data),
                'unfilled_jobs': len(self.job_data[~self.job_data['filled']]),
                'fill_rate': float(self.job_data['filled'].mean()) if len(self.job_data) > 0 else 0.0
            },
            'category_metrics': {},  # Detailed metrics by category
            'skill_distribution': {},  # Distribution of skills across jobs
            'budget_analysis': {},  # Budget analysis by category
            'temporal_patterns': {}  # Temporal patterns by category
        }
        
        # Analyze each category
        for category in self.job_data['category'].unique():
            category_jobs = self.job_data[self.job_data['category'] == category]
            # Basic category metrics
            # Handle empty slices to avoid numpy warnings
            metrics = {
                'total_jobs': len(category_jobs),
                'unfilled_jobs': len(category_jobs[~category_jobs['filled']]),
                'fill_rate': 0.0,
                'avg_bids': 0.0,
                'unfilled_rate_difference': 0.0,
                'unfilled_skill_complexity': 0.0,
                'unfilled_avg_rate': 0.0,
                'unfilled_avg_skills': 0.0
            }
            
            # Calculate basic metrics with explicit error handling
            try:
                if len(category_jobs) > 0:
                    metrics['fill_rate'] = float(category_jobs['filled'].mean())
                    metrics['avg_bids'] = float(category_jobs['num_bids'].mean())
            except (TypeError, ValueError):
                pass  # Keep default 0.0
            
            # Analyze unfilled jobs in this category
            unfilled_jobs = category_jobs[~category_jobs['filled']]
            if len(unfilled_jobs) > 0:
                try:
                    # Calculate rates with explicit error handling
                    category_avg_rate = float(category_jobs['budget_amount'].mean()) if len(category_jobs) > 0 else 0.0
                    unfilled_avg_rate = float(unfilled_jobs['budget_amount'].mean())
                    
                    if category_avg_rate > 0:
                        rate_difference = ((unfilled_avg_rate - category_avg_rate) / category_avg_rate * 100)
                    else:
                        rate_difference = 0.0
                        
                    # Calculate skill metrics with explicit error handling
                    avg_skills_required = float(unfilled_jobs['skills_required'].apply(len).mean())
                    category_avg_skills = float(category_jobs['skills_required'].apply(len).mean()) if len(category_jobs) > 0 else 0.0
                    skill_difference = avg_skills_required - category_avg_skills
                    
                    metrics.update({
                        'unfilled_rate_difference': rate_difference,  # % difference from category average
                        'unfilled_skill_complexity': skill_difference,  # Difference in required skills
                        'unfilled_avg_rate': unfilled_avg_rate,
                        'unfilled_avg_skills': avg_skills_required
                    })
                except (TypeError, ValueError):
                    pass  # Keep default 0.0
            
            unfilled_analysis['category_metrics'][category] = metrics
            
            # Analyze skill distribution
            if len(category_jobs) > 0:
                # Initialize with default values
                skill_metrics = {
                    'avg_required_skills': 0.0,
                    'skill_variety': 0
                }
                
                try:
                    # Calculate average required skills with explicit error handling
                    skill_metrics['avg_required_skills'] = float(category_jobs['skills_required'].apply(len).mean())
                    
                    # Calculate skill variety with explicit error handling
                    all_skills = set()
                    for skills in category_jobs['skills_required']:
                        if isinstance(skills, (list, tuple)):
                            all_skills.update(skills)
                    skill_metrics['skill_variety'] = len(all_skills)
                except (TypeError, ValueError):
                    pass  # Keep default values
                
                unfilled_analysis['skill_distribution'][category] = skill_metrics
            
            # Analyze budget patterns
            if len(category_jobs) > 0:
                filled_jobs = category_jobs[category_jobs['filled']]
                unfilled_jobs = category_jobs[~category_jobs['filled']]
                
                # Initialize with default values
                budget_metrics = {
                    'category_avg_rate': 0.0,
                    'filled_avg_rate': 0.0,
                    'unfilled_avg_rate': 0.0,
                    'budget_distribution': {
                        'below_market': 0.0,
                        'at_market': 0.0,
                        'above_market': 0.0
                    }
                }
                
                try:
                    # Calculate average rates with explicit error handling
                    budget_metrics['category_avg_rate'] = float(category_jobs['budget_amount'].mean())
                    if len(filled_jobs) > 0:
                        budget_metrics['filled_avg_rate'] = float(filled_jobs['budget_amount'].mean())
                    if len(unfilled_jobs) > 0:
                        budget_metrics['unfilled_avg_rate'] = float(unfilled_jobs['budget_amount'].mean())
                    
                    # Calculate budget distribution relative to category average
                    category_avg = budget_metrics['category_avg_rate']
                    if category_avg > 0 and len(unfilled_jobs) > 0:
                        below_market = len(unfilled_jobs[unfilled_jobs['budget_amount'] < category_avg * 0.8])
                        at_market = len(unfilled_jobs[(unfilled_jobs['budget_amount'] >= category_avg * 0.8) & (unfilled_jobs['budget_amount'] <= category_avg * 1.2)])
                        above_market = len(unfilled_jobs[unfilled_jobs['budget_amount'] > category_avg * 1.2])
                        total = len(unfilled_jobs)
                        
                        if total > 0:
                            budget_metrics['budget_distribution'].update({
                                'below_market': float(below_market) / total,
                                'at_market': float(at_market) / total,
                                'above_market': float(above_market) / total
                            })
                except (TypeError, ValueError):
                    pass  # Keep default values
                
                unfilled_analysis['budget_analysis'][category] = budget_metrics
            
            # Analyze temporal patterns
            if len(category_jobs) > 0:
                try:
                    # Convert timestamps to round numbers
                    timestamps = pd.to_datetime(category_jobs['posted_time'])
                    # Create a copy to avoid SettingWithCopyWarning
                    category_jobs = category_jobs.copy()
                    category_jobs.loc[:, 'round_num'] = ((timestamps - timestamps.min()).dt.total_seconds() / 3600).astype(int)
                    
                    # Initialize metrics with default values
                    round_metrics = {}
                    for metric in ['filled', 'num_bids', 'budget_amount']:
                        round_metrics[metric] = {}
                    
                    # Calculate metrics by round with explicit error handling
                    grouped = category_jobs.groupby('round_num')
                    for round_num, group in grouped:
                        for metric in ['filled', 'num_bids', 'budget_amount']:
                            try:
                                if len(group) > 0:
                                    round_metrics[metric][round_num] = float(group[metric].mean())
                                else:
                                    round_metrics[metric][round_num] = 0.0
                            except (TypeError, ValueError):
                                round_metrics[metric][round_num] = 0.0
                    
                    # Calculate early vs late round stats
                    mid_point = category_jobs['round_num'].median()
                    early_jobs = category_jobs[category_jobs['round_num'] < mid_point]
                    late_jobs = category_jobs[category_jobs['round_num'] >= mid_point]
                    
                    # Initialize temporal metrics with default values
                    temporal_metrics = {
                        'metrics_by_round': round_metrics,
                        'early_fill_rate': 0.0,
                        'late_fill_rate': 0.0,
                        'early_avg_bids': 0.0,
                        'late_avg_bids': 0.0
                    }
                    
                    # Calculate early/late metrics with explicit error handling
                    if len(early_jobs) > 0:
                        try:
                            temporal_metrics['early_fill_rate'] = float(early_jobs['filled'].mean())
                            temporal_metrics['early_avg_bids'] = float(early_jobs['num_bids'].mean())
                        except (TypeError, ValueError):
                            pass  # Keep default values
                    
                    if len(late_jobs) > 0:
                        try:
                            temporal_metrics['late_fill_rate'] = float(late_jobs['filled'].mean())
                            temporal_metrics['late_avg_bids'] = float(late_jobs['num_bids'].mean())
                        except (TypeError, ValueError):
                            pass  # Keep default values
                    
                    unfilled_analysis['temporal_patterns'][category] = temporal_metrics
                except Exception:
                    # If any error occurs during temporal analysis, use default values
                    unfilled_analysis['temporal_patterns'][category] = {
                        'metrics_by_round': {
                            'filled': {},
                            'num_bids': {},
                            'budget_amount': {}
                        },
                        'early_fill_rate': 0.0,
                        'late_fill_rate': 0.0,
                        'early_avg_bids': 0.0,
                        'late_avg_bids': 0.0
                    }
        
        return unfilled_analysis

    def analyze_economic_patterns(self):
        """Analyze emergent economic behavior and market dynamics"""
        if self.job_data is None or self.bid_data is None or self.freelancer_states is None:
            return None
            
        # Convert timestamps to round numbers
        timestamps = pd.to_datetime(self.job_data['posted_time'])
        self.min_time = timestamps.min()  # Store as instance variable
        self.job_data['round_num'] = ((timestamps - self.min_time).dt.total_seconds() / 3600).astype(int)
        
        economic_analysis = {
            'inequality_metrics': self._analyze_inequality(),
            'strategy_evolution': self._analyze_strategy_evolution(),
            'market_dynamics': self._analyze_market_dynamics()
        }
        
        return economic_analysis
        
    def _analyze_inequality(self) -> Dict:
        """Analyze economic inequality patterns"""
        # Group freelancers by characteristics
        freelancer_groups = {}
        for _, freelancer in self.freelancer_states.iterrows():
            # Create group key based on relevant characteristics
            experience_level = "new" if freelancer.get('completed_jobs', 0) < 5 else "experienced"
            # Calculate market average rate from available freelancer data
            market_avg_rate = self.freelancer_states.get('min_hourly_rate', 0).mean() if len(self.freelancer_states) > 0 else 30.0
            freelancer_rate = freelancer.get('min_hourly_rate', freelancer.get('avg_rate', 0))
            rate_level = "low" if freelancer_rate < market_avg_rate else "high"
            group_key = f"{experience_level}_{rate_level}"
            
            if group_key not in freelancer_groups:
                freelancer_groups[group_key] = []
            freelancer_groups[group_key].append(freelancer)
        
        # Calculate inequality metrics
        inequality_metrics = {}
        for group_key, group in freelancer_groups.items():
            metrics = {
                'bid_count': sum(f.get('total_bids', 0) for f in group),
                'success_rate': sum(f.get('completed_jobs', 0) for f in group) / max(1, sum(f.get('total_bids', 0) for f in group)),
                'avg_earnings': sum(f.get('total_earnings', 0) for f in group) / len(group),
                'market_access': sum(1 for f in group if f.get('total_bids', 0) > 0) / len(group)
            }
            inequality_metrics[group_key] = metrics
        
        return inequality_metrics
        
    def _analyze_strategy_evolution(self) -> Dict:
        """Analyze how strategies evolve over time"""
        # Track strategy changes across rounds
        strategy_evolution = {
            'rate_strategies': {},
            'category_focus': {},
            'adaptation_patterns': {}
        }
        
        # Check if we have bid data for analysis
        if self.bid_data is None or self.bid_data.empty:
            return strategy_evolution
        
        # Analyze rate strategy evolution
        for round_num in self.job_data['round_num'].unique():
            round_bids = self.bid_data[
                pd.to_datetime(self.bid_data['submission_time']) <= 
                (self.min_time + timedelta(hours=int(round_num)))
            ]
            
            # Calculate rate strategies (only if there are bids)
            if len(round_bids) > 0:
                strategy_evolution['rate_strategies'][round_num] = {
                    'avg_bid_rate': float(round_bids['proposed_rate'].mean()),
                    'rate_spread': float(round_bids['proposed_rate'].std()),
                    'competitive_bids': len(round_bids[round_bids['proposed_rate'] < round_bids['proposed_rate'].mean()]) / len(round_bids)
                }
            else:
                strategy_evolution['rate_strategies'][round_num] = {
                    'avg_bid_rate': 0.0,
                    'rate_spread': 0.0,
                    'competitive_bids': 0.0
                }
            
            # Track category focus
            cat_bids = round_bids.groupby('category').size()
            strategy_evolution['category_focus'][round_num] = cat_bids.to_dict()
            
            # Analyze adaptation patterns
            successful_bids = round_bids[round_bids['is_winner']]
            if len(successful_bids) > 0:
                strategy_evolution['adaptation_patterns'][round_num] = {
                    'winning_rate_diff': float((successful_bids['proposed_rate'] - round_bids['proposed_rate'].mean()).mean()),
                    'category_success': successful_bids.groupby('category').size().to_dict()
                }
        
        return strategy_evolution
        
    def _analyze_market_dynamics(self) -> Dict:
        """Analyze market evolution and dynamics"""
        # Group by round
        grouped = self.job_data.groupby('round_num')
        round_metrics = pd.DataFrame()
        
        # Basic market metrics
        for metric in ['budget_amount', 'filled', 'num_bids']:
            round_metrics[metric] = grouped[metric].agg(lambda x: float(x.mean()) if len(x) > 0 else 0.0).fillna(0.0)
        
        # Calculate market efficiency metrics
        market_dynamics = {
            'price_discovery': {
                'avg_budget_trend': round_metrics['budget_amount'].to_dict(),
                'bid_rate_trend': self._calculate_bid_rate_trend(),
                'price_convergence': self._calculate_price_convergence()
            },
            'market_efficiency': {
                'fill_rate_trend': round_metrics['filled'].to_dict(),
                'time_to_fill': self._calculate_time_to_fill(),
                'matching_quality': self._calculate_matching_quality()
            },
            'competition_dynamics': {
                'bid_volume_trend': round_metrics['num_bids'].to_dict(),
                'market_concentration': self._calculate_market_concentration(),
                'entry_exit_patterns': self._calculate_entry_exit_patterns()
            }
        }
        
        return market_dynamics
        
    def _calculate_bid_rate_trend(self) -> Dict:
        """Calculate trends in bid rates"""
        bid_rates = {}
        
        # Check if we have bid data
        if self.bid_data is None or self.bid_data.empty:
            return bid_rates
        
        for round_num in sorted(self.job_data['round_num'].unique()):
            round_bids = self.bid_data[
                pd.to_datetime(self.bid_data['submission_time']) <= 
                (self.min_time + timedelta(hours=int(round_num)))
            ]
            if len(round_bids) > 0:
                bid_rates[round_num] = {
                    'mean': float(round_bids['proposed_rate'].mean()),
                    'median': float(round_bids['proposed_rate'].median()),
                    'std': float(round_bids['proposed_rate'].std())
                }
        return bid_rates
        
    def _calculate_price_convergence(self) -> Dict:
        """Analyze price convergence patterns"""
        convergence_metrics = {}
        
        # Check if we have bid data
        if self.bid_data is None or self.bid_data.empty:
            return convergence_metrics
        
        for category in self.job_data['category'].unique():
            cat_bids = self.bid_data[self.bid_data['category'] == category]
            if len(cat_bids) > 0:
                # Calculate price dispersion over time
                dispersion = cat_bids.groupby(pd.to_datetime(cat_bids['submission_time']).dt.hour)['proposed_rate'].agg(['std', 'mean'])
                if len(dispersion) > 0:
                    convergence_metrics[category] = {
                        'initial_dispersion': float(dispersion['std'].iloc[0]) if not pd.isna(dispersion['std'].iloc[0]) else 0.0,
                        'final_dispersion': float(dispersion['std'].iloc[-1]) if not pd.isna(dispersion['std'].iloc[-1]) else 0.0,
                        'convergence_rate': float((dispersion['std'].iloc[0] - dispersion['std'].iloc[-1]) / dispersion['std'].iloc[0]) if len(dispersion) > 1 and dispersion['std'].iloc[0] > 0 else 0.0
                    }
        return convergence_metrics
        
    def _calculate_time_to_fill(self) -> Dict:
        """Calculate job filling time patterns"""
        fill_times = {}
        
        # Get filled jobs by matching with hiring outcomes
        # Handle both HiringDecision objects and dict formats
        filled_job_ids = set()
        for h in self.hiring_outcomes:
            if hasattr(h, 'job_id') and hasattr(h, 'selected_freelancer'):
                # HiringDecision object
                if h.selected_freelancer:
                    filled_job_ids.add(h.job_id)
            elif isinstance(h, dict):
                # Dictionary format (legacy)
                if h.get('selected_freelancer'):
                    filled_job_ids.add(h['job_id'])
        
        for category in self.job_data['category'].unique():
            cat_jobs = self.job_data[self.job_data['category'] == category]
            filled_jobs = cat_jobs[cat_jobs['id'].isin(filled_job_ids)]
            
            if len(filled_jobs) > 0:
                # Calculate time to fill from job posting to hiring decision
                times = []
                for _, job in filled_jobs.iterrows():
                    job_posted = pd.to_datetime(job['posted_time'])
                    # Find corresponding hiring decision
                    hiring_decision = next((h for h in self.hiring_outcomes if h['job_id'] == job['id']), None)
                    if hiring_decision and hiring_decision.get('timestamp'):
                        hired_time = pd.to_datetime(hiring_decision['timestamp'])
                        time_diff = (hired_time - job_posted).total_seconds() / 3600  # hours
                        times.append(time_diff)
                
                if times:
                    fill_times[category] = {
                        'avg_time': float(np.mean(times)),
                        'min_time': float(np.min(times)),
                        'max_time': float(np.max(times))
                    }
                    
        return fill_times
        
    def _calculate_matching_quality(self) -> Dict:
        """Analyze quality of job-freelancer matches"""
        matching_metrics = {}
        for category in self.job_data['category'].unique():
            cat_jobs = self.job_data[self.job_data['category'] == category]
            if len(cat_jobs) > 0:
                # Calculate skill match scores and success rates
                matching_metrics[category] = {
                    'avg_skill_match': float(cat_jobs['skill_match_score'].mean()),
                    'completion_rate': float(cat_jobs['completed'].mean() if 'completed' in cat_jobs.columns else 0.0),
                    'satisfaction_score': float(cat_jobs['satisfaction_score'].mean() if 'satisfaction_score' in cat_jobs.columns else 0.0)
                }
        return matching_metrics
        
    def _calculate_market_concentration(self) -> Dict:
        """Calculate market concentration metrics"""
        concentration_metrics = {}
        for category in self.job_data['category'].unique():
            cat_bids = self.bid_data[self.bid_data['category'] == category]
            if len(cat_bids) > 0:
                # Calculate Herfindahl-Hirschman Index
                market_shares = cat_bids.groupby('freelancer_id').size() / len(cat_bids)
                hhi = float((market_shares ** 2).sum())
                
                concentration_metrics[category] = {
                    'hhi': hhi,
                    'active_freelancers': len(market_shares),
                    'top_share': float(market_shares.max())
                }
        return concentration_metrics
        
    def _calculate_entry_exit_patterns(self) -> Dict:
        """Analyze market entry and exit patterns"""
        entry_exit = {}
        for round_num in sorted(self.job_data['round_num'].unique()):
            round_bids = self.bid_data[
                pd.to_datetime(self.bid_data['submission_time']) <= 
                (self.min_time + timedelta(hours=int(round_num)))
            ]
            
            # Calculate entry/exit
            current_freelancers = set(round_bids['freelancer_id'])
            if round_num > 0:
                prev_freelancers = set(self.bid_data[
                    pd.to_datetime(self.bid_data['submission_time']) <= 
                    (self.min_time + timedelta(hours=int(round_num-1)))
                ]['freelancer_id'])
                
                entry_exit[round_num] = {
                    'entries': len(current_freelancers - prev_freelancers),
                    'exits': len(prev_freelancers - current_freelancers),
                    'active_freelancers': len(current_freelancers)
                }
            else:
                entry_exit[round_num] = {
                    'entries': len(current_freelancers),
                    'exits': 0,
                    'active_freelancers': len(current_freelancers)
                }
        
        return entry_exit

    def analyze_job_duration_impact(self):
        """Study the impact of job duration on bidding behavior and success rates"""
        if self.job_data is None or self.bid_data is None:
            return None
            
        duration_analysis = {
            'duration_fill_rates': {},  # Fill rate by duration
            'duration_bid_counts': {},  # Number of bids by duration
            'duration_rate_impact': {},  # Average bid rate by duration
            'duration_success_rates': {}  # Job completion success by duration
        }
        
        # Parse timeline to get duration in days
        def parse_timeline(timeline):
            # Example: "2-3 weeks" -> average of 14-21 days
            if not isinstance(timeline, str):
                return 14  # default to 2 weeks if missing
            
            timeline = timeline.lower()
            if 'day' in timeline:
                nums = [int(n) for n in timeline.split() if n.isdigit()]
                return sum(nums) / len(nums) if nums else 7
            elif 'week' in timeline:
                nums = [int(n) for n in timeline.split() if n.isdigit()]
                return (sum(nums) / len(nums) if nums else 2) * 7
            elif 'month' in timeline:
                nums = [int(n) for n in timeline.split() if n.isdigit()]
                return (sum(nums) / len(nums) if nums else 1) * 30
            return 14  # default to 2 weeks
            
        self.job_data['duration_days'] = self.job_data['timeline'].apply(parse_timeline)
        
        # Group jobs by duration
        duration_days = self.job_data['duration_days'].fillna(14)  # Default to 2 weeks if missing
        self.job_data['duration_group'] = pd.cut(duration_days, 
                                               bins=[0, 7, 14, 21, 28, float('inf')],
                                               labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        
        # Calculate metrics by duration group
        # Count bids per job
        if self.bid_data is not None and not self.bid_data.empty:
            job_bid_counts = self.bid_data.groupby('job_id').size()
            self.job_data['num_bids'] = self.job_data['id'].map(job_bid_counts).fillna(0)
        else:
            self.job_data['num_bids'] = 0
        
        # Calculate if job was filled (actually hired someone)
        hired_job_ids = self._get_hired_job_ids()
        if hired_job_ids:
            self.job_data['filled'] = self.job_data['id'].isin(hired_job_ids)
        else:
            # Fallback to old logic if no hiring outcomes available
            self.job_data['filled'] = self.job_data['num_bids'] > 0
        
        # Calculate if job was actually hired (from hiring outcomes)
        # Handle both HiringDecision objects and dict formats
        filled_job_ids = set()
        for h in self.hiring_outcomes:
            if hasattr(h, 'job_id') and hasattr(h, 'selected_freelancer'):
                # HiringDecision object
                if h.selected_freelancer:
                    filled_job_ids.add(h.job_id)
            elif isinstance(h, dict):
                # Dictionary format (legacy)
                if h.get('selected_freelancer'):
                    filled_job_ids.add(h['job_id'])
        self.job_data['hired'] = self.job_data['id'].isin(filled_job_ids)
        
        # Add skill match score (placeholder - can be enhanced later)
        self.job_data['skill_match_score'] = 0.7  # Default moderate match
        
        # Initialize metrics with default values
        duration_metrics = pd.DataFrame(index=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        duration_metrics['id'] = 0
        duration_metrics['filled'] = 0.0
        duration_metrics['num_bids'] = 0.0
        
        try:
            # Calculate metrics with explicit error handling
            grouped = self.job_data.groupby('duration_group', observed=True)
            for duration, group in grouped:
                try:
                    # Count jobs
                    duration_metrics.loc[duration, 'id'] = len(group)
                    
                    # Calculate fill rate
                    if len(group) > 0:
                        try:
                            duration_metrics.loc[duration, 'filled'] = float(group['filled'].mean())
                        except (TypeError, ValueError):
                            pass  # Keep default value
                        
                        # Calculate average bids
                        try:
                            duration_metrics.loc[duration, 'num_bids'] = float(group['num_bids'].mean())
                        except (TypeError, ValueError):
                            pass  # Keep default value
                except Exception:
                    pass  # Keep default values
        except Exception:
            pass  # Keep default values
        
        # Initialize rate analysis with default values
        rate_by_duration = pd.DataFrame(
            0.0,  # Initialize with 0.0 to avoid downcasting warning
            index=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'],
            columns=pd.MultiIndex.from_tuples([('proposed_rate', 'mean'), ('proposed_rate', 'std')])
        )
        
        try:
            # Merge with bid data for rate analysis
            if self.bid_data is not None and not self.bid_data.empty:
                duration_bid_analysis = pd.merge(self.job_data[['id', 'duration_group']], 
                                               self.bid_data[['job_id', 'proposed_rate']], 
                                               left_on='id', right_on='job_id')
            else:
                duration_bid_analysis = pd.DataFrame()
            
            if len(duration_bid_analysis) > 0:
                # Calculate rate metrics with explicit error handling
                grouped = duration_bid_analysis.groupby('duration_group', observed=True)
                for duration, group in grouped:
                    try:
                        if len(group) > 0:
                            # Calculate mean rate
                            try:
                                rate_by_duration.loc[duration, ('proposed_rate', 'mean')] = float(group['proposed_rate'].mean())
                            except (TypeError, ValueError):
                                pass  # Keep default value
                            
                            # Calculate rate std
                            try:
                                rate_by_duration.loc[duration, ('proposed_rate', 'std')] = float(group['proposed_rate'].std())
                            except (TypeError, ValueError):
                                pass  # Keep default value
                    except Exception:
                        pass  # Keep default values
        except Exception:
            pass  # Keep default values
        
        # Format results
        for duration in duration_metrics.index:
            duration_analysis['duration_fill_rates'][duration] = duration_metrics.loc[duration, 'filled']
            duration_analysis['duration_bid_counts'][duration] = duration_metrics.loc[duration, 'num_bids']
            duration_analysis['duration_rate_impact'][duration] = {
                'mean_rate': rate_by_duration.loc[duration, ('proposed_rate', 'mean')],
                'rate_std': rate_by_duration.loc[duration, ('proposed_rate', 'std')]
            }
            duration_analysis['duration_success_rates'][duration] = duration_metrics.loc[duration, 'filled']
            
        return duration_analysis
    

    def perform_statistical_tests(self):
        """Perform statistical tests for paper"""
        results = {}
        
        if self.freelancer_states is not None and len(self.freelancer_states) > 0:
            # Calculate success rate statistics
            success_rates = self.freelancer_states['completed_jobs'] / self.freelancer_states['total_bids']
            results['success_rate_stats'] = {
                'mean': float(success_rates.mean()) if len(success_rates) > 0 else 0.0,
                'std': float(success_rates.std()) if len(success_rates) > 0 else 0.0,
                'min': float(success_rates.min()) if len(success_rates) > 0 else 0.0,
                'max': float(success_rates.max()) if len(success_rates) > 0 else 0.0
            }
        
        return results
    
    def generate_visualizations(self, save_dir: str = "figures"):
        """Generate publication-quality figures"""
        from ..visualization.market_plots import (
            plot_market_overview,
            plot_temporal_patterns,
            plot_bidding_strategies,
            plot_agent_learning
        )
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        if self.job_data is not None and self.bid_data is not None:
            # Add category to bid data if not already present
            if 'category' not in self.bid_data.columns:
                job_categories = self.job_data.set_index('id')['category']
                self.bid_data['category'] = self.bid_data['job_id'].map(job_categories)
            
            # Calculate category metrics
            category_metrics = []
            for category in self.job_data['category'].unique():
                category_jobs = self.job_data[self.job_data['category'] == category]
                category_bids = self.bid_data[self.bid_data['category'] == category]
                
                # Handle empty slices to avoid numpy warnings
                metrics = {
                    'category': category,
                    'fill_rate': float(category_jobs['hired'].mean()) if len(category_jobs) > 0 else 0.0,
                    'avg_bids': float(category_jobs['num_bids'].mean()) if len(category_jobs) > 0 else 0.0,
                    'proposed_rate': category_bids['proposed_rate'] if len(category_bids) > 0 else [],
                    'total_jobs': len(category_jobs),
                    'total_bids': len(category_bids),
                    'total_wins': len(category_bids[category_bids['is_winner']]),
                    'win_rate': float(category_bids['is_winner'].mean()) if len(category_bids) > 0 else 0.0,
                    'avg_bid_rate': float(category_bids['proposed_rate'].mean()) if len(category_bids) > 0 else 0.0,
                    'avg_bid_position': float(category_bids.groupby('job_id').cumcount().mean()) + 1 if len(category_bids) > 0 else 0.0
                }
                category_metrics.append(metrics)
            
            category_df = pd.DataFrame(category_metrics)
            
            # Generate market overview plots
            plot_market_overview(category_df, save_dir)
            
            # Generate temporal pattern plots
            # Split into 4 periods based on job order
            self.job_data['period'] = pd.qcut(range(len(self.job_data)), 4, labels=['Period 1', 'Period 2', 'Period 3', 'Period 4'])
            
            # Calculate temporal metrics
            temporal_metrics = []
            for period in ['Period 1', 'Period 2', 'Period 3', 'Period 4']:
                period_jobs = self.job_data[self.job_data['period'] == period]
                # Initialize with default values
                period_metrics = {
                    'period': period,
                    'efficiency': 0.0,
                    'diversity': 0.0,
                    'adaptations': 0.0,
                    'performance': 0.0
                }
                
                try:
                    if len(period_jobs) > 0:
                        # Calculate efficiency
                        try:
                            period_metrics['efficiency'] = float(period_jobs['filled'].mean())
                        except (TypeError, ValueError):
                            pass  # Keep default value
                            
                        # Calculate diversity
                        try:
                            total_categories = len(self.job_data['category'].unique())
                            period_categories = len(period_jobs['category'].unique())
                            if total_categories > 0:
                                period_metrics['diversity'] = float(period_categories) / total_categories
                        except (TypeError, ValueError):
                            pass  # Keep default value
                            
                        # Calculate adaptations
                        try:
                            period_metrics['adaptations'] = float(period_jobs['num_bids'].mean())
                        except (TypeError, ValueError):
                            pass  # Keep default value
                            
                        # Calculate performance
                        try:
                            period_metrics['performance'] = float(period_jobs['budget_amount'].mean())
                        except (TypeError, ValueError):
                            pass  # Keep default value
                except Exception:
                    pass  # Keep default values
                
                temporal_metrics.append(period_metrics)
            
            # Create DataFrame and ensure numeric values
            temporal_df = pd.DataFrame(temporal_metrics)
            for col in temporal_df.columns:
                if col != 'period':  # Skip non-numeric column
                    temporal_df[col] = pd.to_numeric(temporal_df[col], errors='coerce').fillna(0.0)
            plot_temporal_patterns(temporal_df, save_dir)
            
            # Generate bidding strategy plots
            bidding_data = {
                'category_metrics': category_metrics
            }
            plot_bidding_strategies(bidding_data, save_dir)
            
            # Generate agent learning plots
            learning_data = self.analyze_economic_patterns()
            if learning_data:
                plot_agent_learning(learning_data, save_dir)
        
        logger.info(f"Figures saved to {save_path}")
    
    def generate_summary_report(self):
        """Generate a comprehensive analysis report"""
        report = {
            'simulation_overview': {
                'total_rounds': len(self.round_data) if self.round_data is not None else 0,
                'total_jobs': len(self.job_data) if self.job_data is not None else 0,
                'total_bids': len(self.bid_data) if self.bid_data is not None else 0,
                'total_freelancers': len(self.freelancer_states) if self.freelancer_states is not None else 0
            }
        }
        
        # Market efficiency metrics
        if self.round_data is not None:
            efficiency_data = self.generate_market_efficiency_analysis()
            # Handle empty slices to avoid numpy warnings
            fill_rates = pd.Series(efficiency_data['job_fill_rate']).fillna(0.0)
            rounds = pd.Series(efficiency_data['round']).fillna(0)
            avg_bids = pd.Series(efficiency_data['avg_bids_per_job']).fillna(0.0)
            
            # Calculate metrics with explicit handling of empty data
            try:
                avg_fill_rate = float(fill_rates.mean())
            except (TypeError, ValueError):
                avg_fill_rate = 0.0
                
            try:
                fill_rate_trend = 'increasing' if len(fill_rates) > 1 and fill_rates.corr(rounds) > 0 else 'decreasing'
            except (TypeError, ValueError):
                fill_rate_trend = 'stable'
                
            try:
                avg_competition = float(avg_bids.mean())
            except (TypeError, ValueError):
                avg_competition = 0.0
            
            report['market_efficiency'] = {
                'avg_job_fill_rate': avg_fill_rate,
                'fill_rate_trend': fill_rate_trend,
                'avg_competition': avg_competition
            }
        
        # Persona performance
        persona_perf = self.analyze_persona_performance()
        if persona_perf is not None:
            # Add persona performance metrics directly
            report['persona_analysis'] = persona_perf
        
        # Skill distribution and category coverage
        skill_analysis = self.analyze_skill_distribution()
        if skill_analysis is not None:
            report['skill_analysis'] = skill_analysis
            
        # Job duration impact
        duration_analysis = self.analyze_job_duration_impact()
        if duration_analysis is not None:
            report['duration_analysis'] = duration_analysis
            
        # Job complexity analysis
        complexity_analysis = self.analyze_job_complexity()
        if complexity_analysis is not None:
            report['complexity_analysis'] = complexity_analysis
        
        # Statistical tests
        stats_results = self.perform_statistical_tests()
        report['statistical_tests'] = stats_results
        
        return report

def convert_to_serializable(obj):
    """Convert pandas objects to JSON-serializable format"""
    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.columns, pd.MultiIndex):
            # Flatten MultiIndex columns
            obj.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in obj.columns]
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    else:
        return obj

def main():
    """Run complete analysis pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze marketplace simulation results")
    parser.add_argument(
        "--log-file",
        type=str,
        default="results/simuleval/true_gpt_simulation_20250813_194754.json",
        help="Path to the simulation log file to analyze"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/analysis_results.json",
        help="Path where to save the analysis results"
    )
    
    args = parser.parse_args()
    analyzer = MarketplaceAnalyzer(args.log_file)
    analyzer.load_data()
    
    # Run all analyses
    analyses = {
        'unfilled_jobs': analyzer.analyze_unfilled_jobs(),
        'skill_distribution': analyzer.analyze_skill_distribution(),
        'job_duration': analyzer.analyze_job_duration_impact(),
        'job_complexity': analyzer.analyze_job_complexity(),
        'market_efficiency': convert_to_serializable(analyzer.generate_market_efficiency_analysis()) if analyzer.generate_market_efficiency_analysis() is not None else None,
        'bidding_strategies': convert_to_serializable(analyzer.analyze_bidding_strategies()) if analyzer.analyze_bidding_strategies() is not None else None,
        'agent_learning': analyzer.analyze_agent_learning()
    }
    
    # Save detailed analysis results
    output_file = Path(args.output_file)
    with open(output_file, 'w') as f:
        json.dump(analyses, f, indent=2, default=str)
    
    # Print summary of findings
    print("\n=== Analysis Results Summary ===")
    
    if analyses['unfilled_jobs']:
        print("\nFill Rate Analysis:")
        stats = analyses['unfilled_jobs']['overall_stats']
        print(f"- Overall fill rate: {stats['fill_rate']*100:.1f}% ({stats['total_jobs'] - stats['unfilled_jobs']}/{stats['total_jobs']} jobs)")
        
        print("\nCategory Performance:")
        for cat, cat_stats in analyses['unfilled_jobs']['category_metrics'].items():
            print(f"- {cat} ({cat_stats['total_jobs']} jobs): {cat_stats['fill_rate']*100:.1f}% fill rate ({cat_stats['avg_bids']:.1f} bids/job)")
        
        print("\nBudget Analysis:")
        budget = analyses['unfilled_jobs']['budget_analysis']
        
        for category, stats in budget.items():
            print(f"\n{category}:")
            print(f"- Category avg rate: ${stats['category_avg_rate']:.2f}")
            print(f"- Filled jobs avg rate: ${stats['filled_avg_rate']:.2f}")
            if stats['unfilled_avg_rate'] > 0:
                print(f"- Unfilled jobs avg rate: ${stats['unfilled_avg_rate']:.2f}")
        
        print("\nTemporal Patterns:")
        temporal = analyses['unfilled_jobs']['temporal_patterns']
        for category, stats in temporal.items():
            print(f"\n{category}:")
            print(f"- Early fill rate: {stats['early_fill_rate']*100:.1f}%")
            print(f"- Late fill rate: {stats['late_fill_rate']*100:.1f}%")
            print(f"- Early avg bids: {stats['early_avg_bids']:.1f}")
            print(f"- Late avg bids: {stats['late_avg_bids']:.1f}")
    
    if analyses['skill_distribution']:
        print("\nCategory Coverage:")
        coverage = analyses['skill_distribution']['category_coverage']
        for category, stats in coverage.items():
            if stats['total_jobs'] > 0:
                print(f"\n{category}:")
                print(f"- Total jobs: {stats['total_jobs']}")
                print(f"- Total freelancers: {stats['total_freelancers']}")
                print(f"- Avg bids per job: {stats['avg_bids_per_job']:.1f}")
                print(f"- Fill rate: {stats['fill_rate']*100:.1f}%")
                print(f"- Avg job rate: ${stats['avg_job_rate']:.2f}")
    
    if analyses['job_duration']:
        print("\nJob Duration Impact:")
        fill_rates = analyses['job_duration']['duration_fill_rates']
        best_duration = max(fill_rates.items(), key=lambda x: x[1])
        print(f"- Best performing duration: {best_duration[0]} ({best_duration[1]:.1%} fill rate)")
        
        # Show bid patterns
        bid_counts = analyses['job_duration']['duration_bid_counts']
        most_bids = max(bid_counts.items(), key=lambda x: x[1])
        print(f"- Most competitive duration: {most_bids[0]} ({most_bids[1]:.1f} bids on average)")
    
    if analyses['job_complexity']:
        print("\nJob Complexity Analysis:")
        complexity = analyses['job_complexity']['category_complexity']
        for category, stats in complexity.items():
            print(f"\n{category}:")
            print(f"- Avg complexity: {stats['avg_complexity']:.2f}")
            print(f"- Avg budget factor: {stats['avg_budget_factor']:.2f}")
            print(f"- Avg skill factor: {stats['avg_skill_factor']:.2f}")
            print(f"- Avg competition factor: {stats['avg_competition_factor']:.2f}")
            print("Fill rates by complexity:")
            for level, rate in stats['fill_rate_by_complexity'].items():
                print(f"  - {level}: {rate*100:.1f}%")
    
    if analyses['agent_learning']:
        print("\nAgent Learning Analysis:")
        # Analyze freelancer progression
        freelancers = analyses['agent_learning']['freelancer_progression']
        if freelancers:
            win_rate_changes = []
            rate_changes = []
            category_expansions = []
            
            for metrics in freelancers.values():
                if metrics['win_rate_progression']:
                    # Calculate win rate trend
                    win_rates = metrics['win_rate_progression']
                    if len(win_rates) >= 2:
                        win_rate_changes.append(win_rates[-1] - win_rates[0])
                
                if metrics['rate_adaptation']:
                    # Calculate rate adaptation trend
                    rates = metrics['rate_adaptation']
                    if len(rates) >= 2:
                        initial_rate = rates[0]
                        final_rate = rates[-1]
                        rate_changes.append(final_rate - initial_rate)
                
                if metrics['category_expansion']:
                    # Calculate category expansion
                    expansions = metrics['category_expansion']
                    if len(expansions) >= 2:
                        category_expansions.append(expansions[-1] - expansions[0])
            
            # Handle empty slices to avoid numpy warnings
            if win_rate_changes:
                try:
                    avg_win_rate_change = float(sum(win_rate_changes)) / len(win_rate_changes)
                except (ZeroDivisionError, TypeError):
                    avg_win_rate_change = 0.0
                print(f"- Win rate progression: {avg_win_rate_change*100:.1f}% (from first to last bid)")
            
            if rate_changes:
                try:
                    avg_rate_change = float(sum(rate_changes)) / len(rate_changes)
                except (ZeroDivisionError, TypeError):
                    avg_rate_change = 0.0
                print(f"- Rate adaptation: ${avg_rate_change:.2f} average change")
                
                # Show rate change distribution
                positive_changes = sum(1 for change in rate_changes if change > 0)
                negative_changes = sum(1 for change in rate_changes if change < 0)
                print(f"  • {positive_changes} freelancers increased rates")
                print(f"  • {negative_changes} freelancers decreased rates")
            
            if category_expansions:
                try:
                    avg_expansion = float(sum(category_expansions)) / len(category_expansions)
                    max_expansion = max(category_expansions)
                except (ZeroDivisionError, TypeError, ValueError):
                    avg_expansion = 0.0
                    max_expansion = 0
                print(f"- Category expansion: {avg_expansion:.1f} new categories on average")
                print(f"  • Most diverse freelancer: {max_expansion} new categories")
        
        # Analyze client adaptation
        clients = analyses['agent_learning']['client_adaptation']
        if clients:
            avg_budget_change = []
            avg_fill_rate_change = []
            for metrics in clients.values():
                if metrics['budget_adaptation']:
                    avg_budget_change.append(metrics['budget_adaptation'][-1] - metrics['budget_adaptation'][0])
                if metrics['fill_rate_progression']:
                    avg_fill_rate_change.append(metrics['fill_rate_progression'][-1] - metrics['fill_rate_progression'][0])
            
            # Handle empty slices to avoid numpy warnings
            if avg_budget_change:
                try:
                    avg_change = float(sum(avg_budget_change)) / len(avg_budget_change)
                except (ZeroDivisionError, TypeError):
                    avg_change = 0.0
                print(f"- Avg budget adjustment: ${avg_change:.2f}")
            if avg_fill_rate_change:
                try:
                    avg_improvement = float(sum(avg_fill_rate_change)) / len(avg_fill_rate_change)
                except (ZeroDivisionError, TypeError):
                    avg_improvement = 0.0
                print(f"- Avg fill rate improvement: {avg_improvement*100:.1f}%")
    
    # Generate plots using the MarketplaceAnalyzer's built-in visualization function
    analyzer.generate_visualizations("results/figures")
    
    print(f"\nDetailed results saved to {output_file}")
    print("Plots saved to results/figures/")

if __name__ == "__main__":
    main()
