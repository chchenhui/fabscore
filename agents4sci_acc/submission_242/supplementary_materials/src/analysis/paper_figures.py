#!/usr/bin/env python3
"""
Generate figures and tables for the SimulEval++ research paper.
"""

import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_simulation_data(results_file="results/simuleval/true_gpt_simulation_20250814_171646.json"):
    """Load simulation results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def load_analysis_data(results_file="results/analysis_results.json"):
    """Load analysis results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_market_overview_plot(analysis_data, output_dir="results/figures"):
    """Create market overview plot showing key metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Marketplace Analysis Overview', fontsize=16)
    
    # Plot 1: Category Performance (Fill Rates)
    if 'skill_analysis' in analysis_data and 'category_coverage' in analysis_data['skill_analysis']:
        categories = analysis_data['skill_analysis']['category_coverage']
        cat_names = list(categories.keys())
        fill_rates = [categories[cat].get('fill_rate', 0) * 100 for cat in cat_names]
        
        # Filter out categories with no activity
        valid_cats = [(name, rate) for name, rate in zip(cat_names, fill_rates) if rate > 0]
        if valid_cats:
            cat_names, fill_rates = zip(*valid_cats)
            
            ax1.bar(range(len(cat_names)), fill_rates, alpha=0.7, color='skyblue')
            ax1.set_title('Job Fill Rate by Category')
            ax1.set_xlabel('Category')
            ax1.set_ylabel('Fill Rate (%)')
            ax1.set_xticks(range(len(cat_names)))
            ax1.set_xticklabels(cat_names, rotation=45, ha='right')
            ax1.set_ylim(0, 110)
            
            # Add value labels
            for i, v in enumerate(fill_rates):
                ax1.text(i, v + 2, f'{v:.0f}%', ha='center', va='bottom')
    
    # Plot 2: Bidding Competition
    if 'skill_analysis' in analysis_data and 'category_coverage' in analysis_data['skill_analysis']:
        categories = analysis_data['skill_analysis']['category_coverage']
        cat_names = list(categories.keys())
        avg_bids = [categories[cat].get('avg_bids_per_job', 0) for cat in cat_names]
        
        # Filter out categories with no activity
        valid_cats = [(name, bids) for name, bids in zip(cat_names, avg_bids) if bids > 0]
        if valid_cats:
            cat_names, avg_bids = zip(*valid_cats)
            
            ax2.bar(range(len(cat_names)), avg_bids, alpha=0.7, color='lightcoral')
            ax2.set_title('Average Bids per Job by Category')
            ax2.set_xlabel('Category')
            ax2.set_ylabel('Avg Bids per Job')
            ax2.set_xticks(range(len(cat_names)))
            ax2.set_xticklabels(cat_names, rotation=45, ha='right')
            
            # Add value labels
            for i, v in enumerate(avg_bids):
                ax2.text(i, v + max(avg_bids) * 0.02, f'{v:.1f}', ha='center', va='bottom')
    
    # Plot 3: Job Rates by Category
    if 'skill_analysis' in analysis_data and 'category_coverage' in analysis_data['skill_analysis']:
        categories = analysis_data['skill_analysis']['category_coverage']
        cat_names = list(categories.keys())
        avg_rates = [categories[cat].get('avg_job_rate', 0) for cat in cat_names]
        
        # Filter out categories with no activity
        valid_cats = [(name, rate) for name, rate in zip(cat_names, avg_rates) if rate > 0]
        if valid_cats:
            cat_names, avg_rates = zip(*valid_cats)
            
            ax3.bar(range(len(cat_names)), avg_rates, alpha=0.7, color='lightgreen')
            ax3.set_title('Average Job Rates by Category')
            ax3.set_xlabel('Category')
            ax3.set_ylabel('Average Rate ($)')
            ax3.set_xticks(range(len(cat_names)))
            ax3.set_xticklabels(cat_names, rotation=45, ha='right')
            
            # Add value labels
            for i, v in enumerate(avg_rates):
                ax3.text(i, v + max(avg_rates) * 0.02, f'${v:.0f}', ha='center', va='bottom')
    
    # Plot 4: Overall Market Metrics
    if 'simulation_overview' in analysis_data:
        overview = analysis_data['simulation_overview']
        metrics = ['Freelancers', 'Jobs', 'Bids', 'Rounds']
        values = [
            overview.get('total_freelancers', 0),
            overview.get('total_jobs', 0), 
            overview.get('total_bids', 0),
            overview.get('total_rounds', 0)
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax4.bar(metrics, values, alpha=0.7, color=colors)
        ax4.set_title('Simulation Overview')
        ax4.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                    str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/market_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_expertise_skills_plot(simulation_data):
    """Create expertise and skills overview plot."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Expertise and Skills Overview', fontsize=16)
    
    # Extract freelancer data
    freelancers = simulation_data.get('freelancer_profiles', {})
    
    if not freelancers:
        logger.warning("No freelancer profiles found in simulation data")
        # Still return the figure with "No data" messages
        for ax in [ax1, ax2, ax3]:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        return fig
    
    # Plot 1: Skills distribution
    all_skills = []
    all_rates = []
    all_categories = []
    
    for fid, freelancer in freelancers.items():
        skills = freelancer.get('skills', [])
        rate = freelancer.get('min_hourly_rate', 0)
        category = freelancer.get('category', 'Unknown')
        
        all_skills.extend(skills)
        all_rates.append(rate)
        all_categories.append(category)
    
    logger.info(f"Processing {len(freelancers)} freelancers, {len(all_skills)} total skills")
    
    if all_skills:
        skill_counts = pd.Series(all_skills).value_counts().head(10)
        ax1.bar(range(len(skill_counts)), skill_counts.values, alpha=0.7, color='skyblue')
        ax1.set_title('Top 10 Skills Distribution')
        ax1.set_xlabel('Skills')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(skill_counts)))
        ax1.set_xticklabels(skill_counts.index, rotation=45, ha='right')
    else:
        ax1.text(0.5, 0.5, 'No skills data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Top 10 Skills Distribution')
    
    # Plot 2: Rate distribution
    if all_rates:
        ax2.hist(all_rates, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(np.mean(all_rates), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: ${np.mean(all_rates):.0f}')
        ax2.set_title('Hourly Rate Distribution')
        ax2.set_xlabel('Hourly Rate ($)')
        ax2.set_ylabel('Number of Freelancers')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No rate data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Hourly Rate Distribution')
    
    # Plot 3: Experience levels (based on rates)
    if all_rates:
        # Categorize by rate ranges
        experience_levels = []
        for rate in all_rates:
            if rate < 30:
                experience_levels.append('Entry Level (<$30)')
            elif rate < 60:
                experience_levels.append('Mid Level ($30-60)')
            else:
                experience_levels.append('Senior Level (>$60)')
        
        exp_counts = pd.Series(experience_levels).value_counts()
        colors = ['lightgreen', 'gold', 'lightcoral']
        ax3.bar(exp_counts.index, exp_counts.values, color=colors, alpha=0.7)
        ax3.set_title('Experience Level Distribution')
        ax3.set_ylabel('Number of Freelancers')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'No rate data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Experience Level Distribution')
    
    plt.tight_layout()
    # Return the figure for external saving
    return fig

def create_agent_learning_plot(analysis_data):
    """Create a simple agent learning plot from available data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Agent Learning and Adaptation', fontsize=16)
    
    # Plot 1: Market Efficiency Metrics
    if 'market_efficiency' in analysis_data:
        efficiency = analysis_data['market_efficiency']
        metrics = ['Fill Rate', 'Avg Competition']
        values = [
            efficiency.get('avg_job_fill_rate', 0) * 100,
            efficiency.get('avg_competition', 0)
        ]
        
        colors = ['skyblue', 'lightcoral']
        bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_title('Market Efficiency Metrics')
        ax1.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                    f'{value:.1f}', ha='center', va='bottom')
    else:
        ax1.text(0.5, 0.5, 'No market efficiency data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Market Efficiency Metrics')
    
    # Plot 2: Agent Activity Distribution
    if 'persona_analysis' in analysis_data:
        persona = analysis_data['persona_analysis']
        total_freelancers = analysis_data.get('simulation_overview', {}).get('total_freelancers', 1)
        
        activity_metrics = {
            'Avg Bids/Freelancer': persona.get('avg_bids_per_freelancer', 0),
            'Success Rate': persona.get('success_rate', 0) * 100,
            'Total Bids': persona.get('total_bids', 0)
        }
        
        # Normalize for better visualization
        normalized_values = [
            activity_metrics['Avg Bids/Freelancer'] * 10,  # Scale up for visibility
            activity_metrics['Success Rate'],
            activity_metrics['Total Bids'] / total_freelancers  # Per freelancer
        ]
        
        colors = ['gold', 'lightgreen', 'lightblue']
        bars = ax2.bar(activity_metrics.keys(), normalized_values, color=colors, alpha=0.7)
        ax2.set_title('Agent Activity Patterns')
        ax2.set_ylabel('Normalized Value')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add actual value labels
        original_values = list(activity_metrics.values())
        for bar, value in zip(bars, original_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(normalized_values) * 0.01,
                    f'{value:.1f}', ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'No persona analysis data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Agent Activity Patterns')
    
    plt.tight_layout()
    # Return the figure for external saving
    return fig

def create_interaction_metrics_plot(data):
    """Create a plot showing system-level interaction metrics."""
    jobs = data['all_jobs']
    hiring_outcomes = data['hiring_outcomes']
    
    # Categorize jobs
    job_categories = {}
    for job in jobs:
        title = job['title'].lower()
        if 'logo' in title or 'design' in title:
            category = 'Design/Creative'
        elif 'market research' in title or 'research' in title:
            category = 'Market Research'
        elif 'content' in title or 'writing' in title:
            category = 'Content Writing'
        elif 'strategy' in title or 'consultant' in title:
            category = 'Business Strategy'
        else:
            category = 'Other'
        
        if category not in job_categories:
            job_categories[category] = {'posted': 0, 'filled': 0, 'total_budget': 0}
        
        job_categories[category]['posted'] += 1
        job_categories[category]['total_budget'] += job['budget_amount']
    
    # Check which jobs were filled
    for outcome in hiring_outcomes:
        if outcome['selected_freelancer']:
            job_title = outcome['job'].split("title='")[1].split("'")[0].lower()
            if 'logo' in job_title or 'design' in job_title:
                category = 'Design/Creative'
            elif 'market research' in job_title or 'research' in job_title:
                category = 'Market Research'
            elif 'content' in job_title or 'writing' in job_title:
                category = 'Content Writing'
            elif 'strategy' in job_title or 'consultant' in job_title:
                category = 'Business Strategy'
            else:
                category = 'Other'
            
            if category in job_categories:
                job_categories[category]['filled'] += 1
    
    # Create visualization
    categories = list(job_categories.keys())
    posted = [job_categories[cat]['posted'] for cat in categories]
    filled = [job_categories[cat]['filled'] for cat in categories]
    fill_rates = [(filled[i]/posted[i]*100) if posted[i] > 0 else 0 for i in range(len(posted))]
    avg_budgets = [job_categories[cat]['total_budget']/job_categories[cat]['posted'] 
                   for cat in categories]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Jobs posted vs filled by category
    x_pos = np.arange(len(categories))
    ax1.bar(x_pos - 0.2, posted, 0.4, label='Jobs Posted', alpha=0.8)
    ax1.bar(x_pos + 0.2, filled, 0.4, label='Jobs Filled', alpha=0.8)
    ax1.set_xlabel('Job Category')
    ax1.set_ylabel('Number of Jobs')
    ax1.set_title('Job Market Activity by Category')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    
    # Plot 2: Fill rate and average budget
    ax2_twin = ax2.twinx()
    bars = ax2.bar(x_pos, fill_rates, alpha=0.7, color='green', label='Fill Rate (%)')
    line = ax2_twin.plot(x_pos, avg_budgets, 'ro-', linewidth=2, markersize=8, 
                        color='red', label='Avg Budget ($)')
    
    ax2.set_xlabel('Job Category')
    ax2.set_ylabel('Fill Rate (%)', color='green')
    ax2_twin.set_ylabel('Average Budget ($)', color='red')
    ax2.set_title('Fill Rate and Budget by Category')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.set_ylim(0, 100)
    
    # Add legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('papers/figures/interaction_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_api_usage_chart(data):
    """Create a chart showing API usage efficiency."""
    rounds_data = pd.DataFrame(data['round_data'])
    total_calls = data['api_usage']['total_calls']
    total_tokens = data['api_usage']['total_tokens']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: API calls per round
    ax1.bar(rounds_data['round'], rounds_data['api_calls'], alpha=0.8)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cumulative API Calls')
    ax1.set_title('API Usage Growth Across Rounds')
    ax1.set_xticks(range(1, 9))
    
    # Plot 2: Efficiency metrics
    metrics = ['Total API Calls', 'Tokens Used (K)', 'Jobs Posted', 'Successful Hires']
    values = [total_calls, total_tokens/1000, 16, 4]
    colors = ['blue', 'green', 'orange', 'red']
    
    ax2.bar(metrics, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title('Simulation Efficiency Metrics')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('papers/figures/api_usage.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_paper_figures(simulation_file=None, analysis_file=None, output_dir="results/figures"):
    """Generate essential figures for the paper using specified data files.
    
    Args:
        simulation_file (str): Path to simulation JSON file (for skills/expertise plots)
        analysis_file (str): Path to analysis results JSON file (for metrics plots)
        output_dir (str): Directory to save figures to (default: "results/figures")
    """
    # Create figures directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data with defaults if not specified
    if analysis_file:
        analysis_data = load_analysis_data(analysis_file)
    else:
        analysis_data = load_analysis_data()
    
    if simulation_file:
        simulation_data = load_simulation_data(simulation_file)
    else:
        simulation_data = load_simulation_data()
    
    print(f"🎨 Generating paper figures...")
    if simulation_file:
        print(f"   📁 Using simulation data: {simulation_file}")
    if analysis_file:
        print(f"   📊 Using analysis data: {analysis_file}")
    
    # Generate essential figures
    create_market_overview_plot(analysis_data, output_dir)
    print("✅ Market overview plot created")
    
    fig = create_agent_learning_plot(analysis_data)
    fig.savefig(f'{output_dir}/agent_learning.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Agent learning plot created")
    
    fig = create_expertise_skills_plot(simulation_data)
    fig.savefig(f'{output_dir}/expertise_skills_overview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Expertise skills overview plot created")
    
    print(f"\n📊 Essential figures saved to {output_dir}/")
    print("Figures available:")
    print("- market_overview.png")
    print("- agent_learning.png") 
    print("- expertise_skills_overview.png")
    
    return {
        'market_overview': f'{output_dir}/market_overview.png',
        'agent_learning': f'{output_dir}/agent_learning.png',
        'expertise_skills_overview': f'{output_dir}/expertise_skills_overview.png'
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate paper figures from simulation and analysis data')
    parser.add_argument('--simulation-file', type=str, 
                       help='Path to simulation JSON file (default: results/simuleval/true_gpt_simulation_20250814_171646.json)')
    parser.add_argument('--analysis-file', type=str,
                       help='Path to analysis results JSON file (default: results/analysis_results.json)')
    
    args = parser.parse_args()
    
    generate_paper_figures(
        simulation_file=args.simulation_file,
        analysis_file=args.analysis_file
    )
