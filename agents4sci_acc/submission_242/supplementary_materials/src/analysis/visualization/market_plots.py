"""
Market Visualization Module

This module handles all visualization functionality for the marketplace analysis.
It provides consistent plotting styles and functions for generating figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def setup_figure_dir(base_dir: str = "results/figures") -> Path:
    """Create and return figure directory"""
    fig_dir = Path(base_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir

def plot_market_overview(analysis_data: Dict[str, Dict], output_dir: str = "results/figures"):
    """Create comprehensive market overview plots based on the new analysis structure"""
    fig_dir = setup_figure_dir(output_dir)
    
    # Create multiple figures for different aspects
    # 1. Expertise and Skills Distribution
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('Expertise and Skills Analysis', fontsize=16)
    
    # Plot 1: Expertise areas distribution
    if 'expertise_areas' in analysis_data:
        expertise = pd.Series(analysis_data['expertise_areas'])
        expertise.plot(kind='bar', ax=ax1)
        ax1.set_title('Distribution of Expertise Areas')
        ax1.set_xlabel('Area')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Top skills distribution
    if 'skills' in analysis_data:
        skills = pd.Series(analysis_data['skills']).sort_values(ascending=False)[:10]
        skills.plot(kind='bar', ax=ax2)
        ax2.set_title('Top 10 Skills')
        ax2.set_xlabel('Skill')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Rate distribution
    if 'rate_distribution' in analysis_data:
        rates = pd.Series(analysis_data['rate_distribution'].get('rates', []))
        if not rates.empty:
            sns.histplot(data=rates, ax=ax3, bins=20)
            ax3.axvline(rates.mean(), color='r', linestyle='--', label=f'Mean: ${rates.mean():.2f}')
            ax3.axvline(rates.median(), color='g', linestyle='--', label=f'Median: ${rates.median():.2f}')
            ax3.set_title('Rate Distribution')
            ax3.set_xlabel('Hourly Rate ($)')
            ax3.set_ylabel('Count')
            ax3.legend()
    
    # Plot 4: Project length preferences
    if 'project_preferences' in analysis_data:
        prefs = pd.Series(analysis_data['project_preferences'])
        prefs.plot(kind='pie', ax=ax4, autopct='%1.1f%%')
        ax4.set_title('Project Length Preferences')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'expertise_skills_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Metrics
    if 'performance_metrics' in analysis_data:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig2.suptitle('Performance Analysis', fontsize=16)
        
        metrics = pd.Series(analysis_data['performance_metrics'])
        
        # Plot 1: Success metrics
        success_metrics = metrics[['success_rate', 'hire_rate']].astype(float)
        success_metrics.plot(kind='bar', ax=ax1)
        ax1.set_title('Success Metrics')
        ax1.set_ylabel('Rate')
        ax1.set_ylim(0, 1)
        
        # Plot 2: Activity metrics
        activity_metrics = metrics[['total_bids', 'total_hired', 'completed_jobs', 'active_jobs']].astype(float)
        activity_metrics.plot(kind='bar', ax=ax2)
        ax2.set_title('Activity Metrics')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Bidding Patterns
    if 'bidding_patterns' in analysis_data and 'temporal_patterns' in analysis_data:
        fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig3.suptitle('Bidding Behavior Analysis', fontsize=16)
        
        # Plot 1: Bidding activity distribution
        bid_dist = pd.Series(analysis_data['bidding_patterns'].get('bid_distribution', {}))
        if not bid_dist.empty:
            bid_dist.plot(kind='bar', ax=ax1)
            ax1.set_title('Bids per Freelancer')
            ax1.set_xlabel('Freelancer ID')
            ax1.set_ylabel('Number of Bids')
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Temporal bidding patterns
        temporal = pd.Series(analysis_data['temporal_patterns'].get('hour_distribution', {}))
        if not temporal.empty:
            temporal.plot(kind='line', ax=ax2, marker='o')
            ax2.set_title('Bidding Time Distribution')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Number of Bids')
            ax2.grid(True)
        
        # Plot 3: Bid intervals
        if 'temporal_patterns' in analysis_data:
            intervals = pd.Series({
                'avg': analysis_data['temporal_patterns'].get('avg_interval', 0),
                'min': analysis_data['temporal_patterns'].get('min_interval', 0),
                'max': analysis_data['temporal_patterns'].get('max_interval', 0)
            })
            intervals.plot(kind='bar', ax=ax3)
            ax3.set_title('Bid Interval Statistics')
            ax3.set_ylabel('Minutes')
        
        # Plot 4: Bid bursts
        if 'temporal_patterns' in analysis_data:
            bursts = pd.Series({
                'rapid_bids': analysis_data['temporal_patterns'].get('bid_bursts', 0),
                'total_bids': analysis_data['bidding_patterns'].get('total_bids', 0)
            })
            bursts.plot(kind='bar', ax=ax4)
            ax4.set_title('Bid Burst Analysis')
            ax4.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'bidding_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Geographic Distribution
    if 'geographic_distribution' in analysis_data:
        plt.figure(figsize=(12, 6))
        locations = pd.Series(analysis_data['geographic_distribution'])
        locations.plot(kind='bar')
        plt.title('Geographic Distribution of Freelancers')
        plt.xlabel('Location')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fig_dir / 'geographic_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_client_analysis(client_data: Dict[str, Dict], output_dir: str = "results/figures"):
    """Create comprehensive client analysis plots"""
    fig_dir = setup_figure_dir(output_dir)
    
    # 1. Company Profiles Overview
    if 'company_profiles' in client_data:
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig1.suptitle('Company Profiles Analysis', fontsize=16)
        
        # Plot 1: Focus areas distribution
        if 'focus_areas' in client_data:
            focus = pd.Series(client_data['focus_areas'])
            focus.plot(kind='bar', ax=ax1)
            ax1.set_title('Focus Areas Distribution')
            ax1.set_xlabel('Area')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Company sizes
        if 'company_sizes' in client_data:
            sizes = pd.Series(client_data['company_sizes'])
            sizes.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
            ax2.set_title('Company Size Distribution')
        
        # Plot 3: Budget philosophies
        if 'budget_philosophies' in client_data:
            budgets = pd.Series(client_data['budget_philosophies'])
            budgets.plot(kind='bar', ax=ax3)
            ax3.set_title('Budget Philosophy Distribution')
            ax3.set_xlabel('Philosophy')
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Hiring styles
        if 'hiring_styles' in client_data:
            styles = pd.Series(client_data['hiring_styles'])
            styles.plot(kind='pie', ax=ax4, autopct='%1.1f%%')
            ax4.set_title('Hiring Style Distribution')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'client_profiles.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Job Posting Analysis
    if 'job_posting_patterns' in client_data:
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig2.suptitle('Job Posting Analysis', fontsize=16)
        
        job_data = pd.DataFrame.from_dict(client_data['job_posting_patterns'], orient='index')
        
        # Plot 1: Jobs per client
        if 'total_jobs' in job_data.columns:
            job_data['total_jobs'].plot(kind='bar', ax=ax1)
            ax1.set_title('Jobs Posted per Client')
            ax1.set_xlabel('Client')
            ax1.set_ylabel('Number of Jobs')
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Average budget by client
        if 'avg_budget' in job_data.columns:
            job_data['avg_budget'].plot(kind='bar', ax=ax2)
            ax2.set_title('Average Budget by Client')
            ax2.set_xlabel('Client')
            ax2.set_ylabel('Average Budget ($)')
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Timeline preferences
        if 'timeline_preferences' in job_data.columns:
            timeline_data = pd.DataFrame([d for d in job_data['timeline_preferences'] if isinstance(d, dict)])
            if not timeline_data.empty:
                timeline_data.sum().plot(kind='pie', ax=ax3, autopct='%1.1f%%')
                ax3.set_title('Timeline Preferences')
        
        # Plot 4: Required skills frequency
        if 'required_skills' in job_data.columns:
            skills_data = pd.DataFrame([d for d in job_data['required_skills'] if isinstance(d, dict)])
            if not skills_data.empty:
                skills_sum = skills_data.sum().sort_values(ascending=False)[:10]
                skills_sum.plot(kind='bar', ax=ax4)
                ax4.set_title('Top 10 Required Skills')
                ax4.set_xlabel('Skill')
                ax4.set_ylabel('Frequency')
                ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'job_posting_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Hiring Behavior Analysis
    if 'hiring_behavior' in client_data:
        fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig3.suptitle('Hiring Behavior Analysis', fontsize=16)
        
        hiring_data = pd.DataFrame.from_dict(client_data['hiring_behavior'], orient='index')
        
        # Plot 1: Hiring decisions
        if all(col in hiring_data.columns for col in ['hires', 'rejections']):
            decisions = hiring_data[['hires', 'rejections']]
            decisions.plot(kind='bar', ax=ax1)
            ax1.set_title('Hiring Decisions by Client')
            ax1.set_xlabel('Client')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
        
        # Plot 2: Hire rates
        if 'hire_rate' in hiring_data.columns:
            hiring_data['hire_rate'].plot(kind='bar', ax=ax2)
            ax2.set_title('Hire Rate by Client')
            ax2.set_xlabel('Client')
            ax2.set_ylabel('Rate')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Average accepted rates
        if 'avg_accepted_rate' in hiring_data.columns:
            hiring_data['avg_accepted_rate'].plot(kind='bar', ax=ax3)
            ax3.set_title('Average Accepted Rate by Client')
            ax3.set_xlabel('Client')
            ax3.set_ylabel('Rate ($)')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Total decisions vs hires
        if all(col in hiring_data.columns for col in ['total_decisions', 'hires']):
            comparison = hiring_data[['total_decisions', 'hires']]
            comparison.plot(kind='bar', ax=ax4)
            ax4.set_title('Total Decisions vs Hires')
            ax4.set_xlabel('Client')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'hiring_behavior.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Background Themes Analysis
    if 'background_themes' in client_data:
        plt.figure(figsize=(12, 6))
        themes = pd.Series(client_data['background_themes'])
        themes.plot(kind='bar')
        plt.title('Client Background Themes')
        plt.xlabel('Theme')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fig_dir / 'background_themes.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_bidding_strategies(bidding_data: Dict[str, Dict], output_dir: str = "results/figures"):
    """Create comprehensive bidding strategy analysis plots"""
    fig_dir = setup_figure_dir(output_dir)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bidding Strategy Analysis', fontsize=16)
    
    # Plot 1: Success rates by category
    if 'category_metrics' in bidding_data:
        categories = []
        success_rates = []
        for cat in bidding_data['category_metrics']:
            categories.append(cat['category'])
            success_rates.append(cat.get('win_rate', 0))
        
        ax1.bar(categories, success_rates)
        ax1.set_title('Success Rates by Category')
        ax1.set_xlabel('Category')
        ax1.set_ylabel('Success Rate')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average bid rates
    if 'category_metrics' in bidding_data:
        avg_rates = [cat.get('avg_bid_rate', 0) for cat in bidding_data['category_metrics']]
        ax2.bar(categories, avg_rates, color='lightcoral')
        ax2.set_title('Average Bid Rates by Category')
        ax2.set_xlabel('Category')
        ax2.set_ylabel('Average Rate ($)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bid timing analysis
    if 'category_metrics' in bidding_data:
        avg_positions = [cat.get('avg_bid_position', 0) for cat in bidding_data['category_metrics']]
        ax3.bar(categories, avg_positions, color='lightgreen')
        ax3.set_title('Average Bid Position by Category')
        ax3.set_xlabel('Category')
        ax3.set_ylabel('Average Position')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Total bids vs wins
    if 'category_metrics' in bidding_data:
        total_bids = [cat.get('total_bids', 0) for cat in bidding_data['category_metrics']]
        total_wins = [cat.get('total_wins', 0) for cat in bidding_data['category_metrics']]
        
        x = range(len(categories))
        width = 0.35
        
        ax4.bar([i - width/2 for i in x], total_bids, width, label='Total Bids', color='skyblue')
        ax4.bar([i + width/2 for i in x], total_wins, width, label='Wins', color='lightcoral')
        ax4.set_title('Bids vs Wins by Category')
        ax4.set_xlabel('Category')
        ax4.set_ylabel('Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'bidding_strategies.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_agent_learning(learning_data: Dict[str, Dict], output_dir: str = "results/figures"):
    """Create agent learning and adaptation analysis plots"""
    fig_dir = setup_figure_dir(output_dir)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Agent Learning Analysis', fontsize=16)
    
    # Plot 1: Market evolution trends
    if 'market_evolution' in learning_data:
        evolution = learning_data['market_evolution']
        rounds = list(range(len(evolution['avg_budget_trend'])))
        
        ax1.plot(rounds, list(evolution['avg_budget_trend'].values()), label='Avg Budget', color='skyblue')
        ax1.plot(rounds, list(evolution['fill_rate_trend'].values()), label='Fill Rate', color='lightcoral')
        ax1.plot(rounds, list(evolution['competition_trend'].values()), label='Competition', color='lightgreen')
        ax1.set_title('Market Evolution Trends')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning progress by topic
    if 'learning_progress' in learning_data:
        topics = []
        progress = []
        for topic, metrics in learning_data['learning_progress'].items():
            topics.append(topic)
            progress.append(metrics.get('success_probability', 0))
        
        ax2.bar(topics, progress, color='lightcoral')
        ax2.set_title('Learning Progress by Topic')
        ax2.set_xlabel('Topic')
        ax2.set_ylabel('Success Probability')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Adaptation metrics
    if 'adaptation_metrics' in learning_data:
        metrics = learning_data['adaptation_metrics']
        metric_names = list(metrics.keys())
        metric_values = [metrics[m] for m in metric_names]
        
        ax3.bar(metric_names, metric_values, color='lightgreen')
        ax3.set_title('Adaptation Metrics')
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance improvement
    if 'performance_improvement' in learning_data:
        improvement = learning_data['performance_improvement']
        metrics = list(improvement.keys())
        initial = [improvement[m].get('initial', 0) for m in metrics]
        final = [improvement[m].get('final', 0) for m in metrics]
        
        x = range(len(metrics))
        width = 0.35
        
        ax4.bar([i - width/2 for i in x], initial, width, label='Initial', color='skyblue')
        ax4.bar([i + width/2 for i in x], final, width, label='Final', color='lightcoral')
        ax4.set_title('Performance Improvement')
        ax4.set_xlabel('Metric')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'agent_learning.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_temporal_patterns(temporal_data: Dict[str, Dict], output_dir: str = "results/figures"):
    """Create temporal pattern plots based on the new analysis structure"""
    fig_dir = setup_figure_dir(output_dir)
    
    # 1. Bidding Temporal Patterns
    if 'temporal_patterns' in temporal_data:
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig1.suptitle('Temporal Bidding Patterns', fontsize=16)
        
        patterns = temporal_data['temporal_patterns']
        
        # Plot 1: Hour distribution
        if 'hour_distribution' in patterns:
            hours = pd.Series(patterns['hour_distribution'])
            hours.plot(kind='line', marker='o', ax=ax1)
            ax1.set_title('Bidding Activity by Hour')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Number of Bids')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bid intervals
        if all(key in patterns for key in ['avg_interval', 'min_interval', 'max_interval']):
            intervals = pd.Series({
                'Average': patterns['avg_interval'],
                'Minimum': patterns['min_interval'],
                'Maximum': patterns['max_interval']
            })
            intervals.plot(kind='bar', ax=ax2)
            ax2.set_title('Bid Interval Statistics')
            ax2.set_ylabel('Minutes')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Bid bursts
        if 'bid_bursts' in patterns:
            bursts = pd.Series({
                'Rapid Bids (<5min)': patterns['bid_bursts'],
                'Other Bids': sum(patterns.get('hour_distribution', {}).values()) - patterns['bid_bursts']
            })
            bursts.plot(kind='pie', ax=ax3, autopct='%1.1f%%')
            ax3.set_title('Bid Burst Analysis')
        
        # Plot 4: Cumulative bids over time
        if 'hour_distribution' in patterns:
            hours = pd.Series(patterns['hour_distribution']).sort_index()
            cumulative = hours.cumsum()
            cumulative.plot(kind='line', ax=ax4)
            ax4.set_title('Cumulative Bids Over Time')
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('Total Bids')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'temporal_bidding_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Success Rate Evolution
    if 'success_factors' in temporal_data:
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig2.suptitle('Success Rate Evolution', fontsize=16)
        
        factors = temporal_data['success_factors']
        
        # Plot 1: Rate impact
        if 'rate_impact' in factors:
            rate_data = pd.Series(factors['rate_impact'].get('successful_rates', {}))
            if not rate_data.empty:
                rate_data.plot(kind='bar', ax=ax1)
                ax1.set_title('Rate Impact on Success')
                ax1.set_ylabel('Rate ($)')
                ax1.grid(True, alpha=0.3)
        
        # Plot 2: Budget ratio distribution
        if 'rate_impact' in factors and 'budget_ratios' in factors['rate_impact']:
            ratios = pd.Series(factors['rate_impact']['budget_ratios'])
            ratios.plot(kind='bar', ax=ax2)
            ax2.set_title('Budget Ratio Distribution')
            ax2.set_ylabel('Ratio')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Message characteristics
        if 'message_characteristics' in factors:
            msg_stats = pd.Series(factors['message_characteristics'].get('length_stats', {}))
            if not msg_stats.empty:
                msg_stats.plot(kind='bar', ax=ax3)
                ax3.set_title('Message Length Statistics')
                ax3.set_ylabel('Word Count')
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Skill alignment
        if 'skill_alignment' in factors:
            alignment = pd.Series(factors['skill_alignment'])
            if not alignment.empty:
                alignment.plot(kind='bar', ax=ax4)
                ax4.set_title('Skill Alignment Metrics')
                ax4.set_ylabel('Score')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'success_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Competition Analysis
    if 'competition_analysis' in temporal_data:
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig3.suptitle('Competition Analysis', fontsize=16)
        
        competition = temporal_data['competition_analysis']
        
        # Plot 1: Competition stats
        if 'competition_stats' in competition:
            stats = pd.Series(competition['competition_stats'])
            stats.plot(kind='bar', ax=ax1)
            ax1.set_title('Competition Statistics')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rate dynamics
        if 'rate_dynamics' in competition:
            dynamics = pd.Series(competition['rate_dynamics'])
            dynamics.plot(kind='bar', ax=ax2)
            ax2.set_title('Rate Spread Dynamics')
            ax2.set_ylabel('Rate Spread ($)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'competition_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Message Elements Analysis
    if 'success_factors' in temporal_data and 'message_characteristics' in temporal_data['success_factors']:
        msg_data = temporal_data['success_factors']['message_characteristics']
        if 'key_elements' in msg_data:
            plt.figure(figsize=(12, 6))
            elements = pd.Series(msg_data['key_elements'])
            elements.plot(kind='bar')
            plt.title('Key Message Elements in Successful Bids')
            plt.xlabel('Element')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / 'message_elements.png', dpi=300, bbox_inches='tight')
            plt.close()

def plot_agent_performance(agent_metrics: Dict[str, Dict], output_dir: str = "results/figures"):
    """Create agent performance comparison plots"""
    fig_dir = setup_figure_dir(output_dir)
    
    # Prepare data
    agent_types = list(agent_metrics.keys())
    # Handle empty slices and ensure numeric values
    metrics = {
        'response_rate': [float(agent_metrics[t].get('response_rate', 0)) for t in agent_types],
        'task_completion': [float(agent_metrics[t].get('task_completion', 0)) for t in agent_types],
        'adaptation_score': [float(agent_metrics[t].get('adaptation_score', 0)) for t in agent_types]
    }
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Response rates
    ax1.bar(agent_types, metrics['response_rate'], color='skyblue')
    ax1.set_title('Response Rates by Agent Type')
    ax1.set_xlabel('Agent Type')
    ax1.set_ylabel('Response Rate (%)')
    ax1.grid(True, alpha=0.3)
    
    # Task completion
    ax2.bar(agent_types, metrics['task_completion'], color='lightcoral')
    ax2.set_title('Task Completion by Agent Type')
    ax2.set_xlabel('Agent Type')
    ax2.set_ylabel('Tasks Completed')
    ax2.grid(True, alpha=0.3)
    
    # Adaptation scores
    ax3.bar(agent_types, metrics['adaptation_score'], color='lightgreen')
    ax3.set_title('Adaptation Scores by Agent Type')
    ax3.set_xlabel('Agent Type')
    ax3.set_ylabel('Adaptation Score')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'agent_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_interaction_metrics(interaction_data: Dict[str, pd.DataFrame], output_dir: str = "results/figures"):
    """Create system-level interaction metrics plots"""
    fig_dir = setup_figure_dir(output_dir)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Interaction patterns over time
    if 'time_series' in interaction_data:
        ts_data = interaction_data['time_series']
        # Handle empty slices to avoid numpy warnings
        interaction_count = pd.to_numeric(ts_data['interaction_count'], errors='coerce').fillna(0.0)
        ax1.plot(ts_data.index, interaction_count, label='Total Interactions')
        if 'successful_interactions' in ts_data.columns:
            successful = pd.to_numeric(ts_data['successful_interactions'], errors='coerce').fillna(0.0)
            ax1.plot(ts_data.index, successful, label='Successful')
        ax1.set_title('Interaction Patterns Over Time')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Number of Interactions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Response distribution
    if 'responses' in interaction_data:
        resp_data = interaction_data['responses']
        sns.boxplot(data=resp_data, ax=ax2)
        ax2.set_title('Response Distribution')
        ax2.set_xlabel('Response Type')
        ax2.set_ylabel('Response Time (s)')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Adaptation trends
    if 'adaptation' in interaction_data:
        adapt_data = interaction_data['adaptation']
        # Handle empty slices to avoid numpy warnings
        adaptation_score = pd.to_numeric(adapt_data['adaptation_score'], errors='coerce').fillna(0.0)
        ax3.plot(adapt_data.index, adaptation_score, color='green')
        ax3.set_title('Adaptation Trends')
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Adaptation Score')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: System performance metrics
    if 'performance' in interaction_data:
        perf_data = interaction_data['performance']
        metrics = ['efficiency', 'reliability', 'responsiveness']
        # Handle empty slices to avoid numpy warnings
        values = [float(perf_data.get(m, 0)) for m in metrics]
        ax4.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax4.set_title('System Performance Metrics')
        ax4.set_ylabel('Score')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'interaction_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_economic_patterns(economic_data: Dict[str, Dict], output_dir: str = "results/figures"):
    """Create plots showing emergent economic behavior and market dynamics"""
    fig_dir = setup_figure_dir(output_dir)
    
    # 1. Inequality Analysis
    if 'inequality_metrics' in economic_data:
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig1.suptitle('Economic Inequality Analysis', fontsize=16)
        
        metrics = economic_data['inequality_metrics']
        groups = list(metrics.keys())
        
        # Plot 1: Bid distribution
        bid_counts = [metrics[g]['bid_count'] for g in groups]
        ax1.bar(groups, bid_counts, color='skyblue')
        ax1.set_title('Bid Distribution by Group')
        ax1.set_xlabel('Group')
        ax1.set_ylabel('Number of Bids')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Success rates
        success_rates = [metrics[g]['success_rate'] for g in groups]
        ax2.bar(groups, success_rates, color='lightcoral')
        ax2.set_title('Success Rates by Group')
        ax2.set_xlabel('Group')
        ax2.set_ylabel('Success Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Average earnings
        earnings = [metrics[g]['avg_earnings'] for g in groups]
        ax3.bar(groups, earnings, color='lightgreen')
        ax3.set_title('Average Earnings by Group')
        ax3.set_xlabel('Group')
        ax3.set_ylabel('Average Earnings ($)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Market access
        access = [metrics[g]['market_access'] for g in groups]
        ax4.bar(groups, access, color='orange')
        ax4.set_title('Market Access by Group')
        ax4.set_xlabel('Group')
        ax4.set_ylabel('Access Rate')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'economic_inequality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Strategy Evolution
    if 'strategy_evolution' in economic_data:
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig2.suptitle('Strategy Evolution Analysis', fontsize=16)
        
        evolution = economic_data['strategy_evolution']
        rounds = sorted(evolution['rate_strategies'].keys())
        
        # Plot 1: Rate strategies
        avg_rates = [evolution['rate_strategies'][r]['avg_bid_rate'] for r in rounds]
        rate_spread = [evolution['rate_strategies'][r]['rate_spread'] for r in rounds]
        competitive = [evolution['rate_strategies'][r]['competitive_bids'] for r in rounds]
        
        ax1.plot(rounds, avg_rates, label='Average Rate', color='skyblue')
        ax1.plot(rounds, rate_spread, label='Rate Spread', color='lightcoral')
        ax1.plot(rounds, competitive, label='Competitive Bids', color='lightgreen')
        ax1.set_title('Rate Strategy Evolution')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Value')
        ax1.legend()
        
        # Plot 2: Category focus
        categories = list(evolution['category_focus'][rounds[0]].keys())
        for cat in categories:
            focus = [evolution['category_focus'][r].get(cat, 0) for r in rounds]
            ax2.plot(rounds, focus, label=cat)
        ax2.set_title('Category Focus Evolution')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Number of Bids')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Adaptation patterns
        if 'adaptation_patterns' in evolution:
            winning_rates = [evolution['adaptation_patterns'].get(r, {}).get('winning_rate_diff', 0) for r in rounds]
            ax3.plot(rounds, winning_rates, color='purple')
            ax3.set_title('Winning Rate Differential')
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Rate Difference ($)')
        
        # Plot 4: Category success evolution
        if 'adaptation_patterns' in evolution:
            for cat in categories:
                success = [evolution['adaptation_patterns'].get(r, {}).get('category_success', {}).get(cat, 0) for r in rounds]
                ax4.plot(rounds, success, label=cat)
            ax4.set_title('Category Success Evolution')
            ax4.set_xlabel('Round')
            ax4.set_ylabel('Successful Bids')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'strategy_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Market Dynamics
    if 'market_dynamics' in economic_data:
        fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig3.suptitle('Market Dynamics Analysis', fontsize=16)
        
        dynamics = economic_data['market_dynamics']
        
        # Plot 1: Price discovery
        rounds = sorted(dynamics['price_discovery']['bid_rate_trend'].keys())
        mean_rates = [dynamics['price_discovery']['bid_rate_trend'][r]['mean'] for r in rounds]
        std_rates = [dynamics['price_discovery']['bid_rate_trend'][r]['std'] for r in rounds]
        
        ax1.plot(rounds, mean_rates, label='Mean Rate', color='skyblue')
        ax1.fill_between(rounds, 
                        [m - s for m, s in zip(mean_rates, std_rates)],
                        [m + s for m, s in zip(mean_rates, std_rates)],
                        alpha=0.2)
        ax1.set_title('Price Discovery Process')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Bid Rate ($)')
        ax1.legend()
        
        # Plot 2: Market efficiency
        fill_rates = list(dynamics['market_efficiency']['fill_rate_trend'].values())
        ax2.plot(rounds, fill_rates, color='lightcoral')
        ax2.set_title('Market Fill Rate Evolution')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Fill Rate')
        
        # Plot 3: Competition dynamics
        categories = list(dynamics['competition_dynamics']['market_concentration'].keys())
        hhi = [dynamics['competition_dynamics']['market_concentration'][cat]['hhi'] for cat in categories]
        ax3.bar(categories, hhi, color='lightgreen')
        ax3.set_title('Market Concentration by Category')
        ax3.set_xlabel('Category')
        ax3.set_ylabel('HHI')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Entry/exit patterns
        rounds = sorted(dynamics['competition_dynamics']['entry_exit_patterns'].keys())
        entries = [dynamics['competition_dynamics']['entry_exit_patterns'][r]['entries'] for r in rounds]
        exits = [dynamics['competition_dynamics']['entry_exit_patterns'][r]['exits'] for r in rounds]
        active = [dynamics['competition_dynamics']['entry_exit_patterns'][r]['active_freelancers'] for r in rounds]
        
        ax4.plot(rounds, entries, label='Entries', color='green')
        ax4.plot(rounds, exits, label='Exits', color='red')
        ax4.plot(rounds, active, label='Active', color='blue')
        ax4.set_title('Market Entry/Exit Dynamics')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Count')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'market_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_temporal_analysis(temporal_data: Dict[str, pd.DataFrame], output_dir: str = "results/figures"):
    """Create temporal analysis plots showing strategy evolution and adaptation"""
    fig_dir = setup_figure_dir(output_dir)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Strategy evolution
    if 'strategies' in temporal_data:
        strat_data = temporal_data['strategies']
        for strategy in strat_data.columns:
            # Handle empty slices to avoid numpy warnings
            strategy_data = pd.to_numeric(strat_data[strategy], errors='coerce').fillna(0.0)
            ax1.plot(strat_data.index, strategy_data, label=strategy)
        ax1.set_title('Strategy Evolution Over Time')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Strategy Usage (%)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Adaptation metrics
    if 'adaptation' in temporal_data:
        adapt_data = temporal_data['adaptation']
        metrics = ['learning_rate', 'strategy_changes', 'performance_improvement']
        for metric in metrics:
            if metric in adapt_data.columns:
                # Handle empty slices to avoid numpy warnings
                metric_data = pd.to_numeric(adapt_data[metric], errors='coerce').fillna(0.0)
                ax2.plot(adapt_data.index, metric_data, label=metric)
        ax2.set_title('Adaptation Metrics Over Time')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance by strategy
    if 'performance' in temporal_data:
        perf_data = temporal_data['performance']
        # Handle empty slices to avoid numpy warnings
        for col in perf_data.columns:
            perf_data[col] = pd.to_numeric(perf_data[col], errors='coerce').fillna(0.0)
        sns.boxplot(data=perf_data, ax=ax3)
        ax3.set_title('Performance by Strategy')
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Performance Score')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative adaptation
    if 'cumulative' in temporal_data:
        cum_data = temporal_data['cumulative']
        # Handle empty slices to avoid numpy warnings
        total_adaptations = pd.to_numeric(cum_data['total_adaptations'], errors='coerce').fillna(0.0)
        ax4.plot(cum_data.index, total_adaptations, color='green')
        ax4.set_title('Cumulative Adaptations')
        ax4.set_xlabel('Time Period')
        ax4.set_ylabel('Total Adaptations')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()


# Multi-run plotting utilities
def setup_figure(figsize: Tuple[int, int] = (12, 8)) -> Tuple:
    """Set up a standard figure with consistent styling"""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def setup_subplots(
    nrows: int, 
    ncols: int, 
    figsize: Tuple[int, int] = (15, 10)
) -> Tuple:
    """Set up subplots with consistent styling"""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows * ncols == 1:
        return fig, axes  # Return single axes object directly
    elif nrows == 1 or ncols == 1:
        return fig, axes.flatten()  # Flatten to 1D array for easy indexing
    else:
        return fig, axes  # Return as 2D array

def save_plot(
    fig, 
    output_path: Path, 
    title: str = None, 
    dpi: int = 300
) -> None:
    """Save plot with consistent formatting"""
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")