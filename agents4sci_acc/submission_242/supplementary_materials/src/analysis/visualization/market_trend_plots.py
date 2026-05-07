"""
Market Trend Visualization Module

Generates plots showing historical trends in market performance and saturation metrics.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json
import logging
from scipy.interpolate import make_interp_spline

logger = logging.getLogger(__name__)

def smooth_line(x, y, num_points=300):
    """Smooth line data using spline interpolation"""
    if len(x) < 3:  # Need at least 3 points for spline
        return x, y
    
    try:
        # Remove any NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = np.array(x)[mask]
        y_clean = np.array(y)[mask]
        
        if len(x_clean) < 3:
            return x, y
        
        # Create spline interpolation
        x_min, x_max = x_clean.min(), x_clean.max()
        x_smooth = np.linspace(x_min, x_max, num_points)
        
        # Use cubic spline with lower degree for fewer oscillations
        spl = make_interp_spline(x_clean, y_clean, k=min(3, len(x_clean)-1))
        y_smooth = spl(x_smooth)
        
        return x_smooth, y_smooth
    except Exception as e:
        logger.warning(f"Smoothing failed: {e}, using original data")
        return x, y

class MarketTrendPlotter:
    """Creates historical trend plots for market metrics"""
    
    def __init__(self, simulation_data: Dict[str, Any]):
        """Initialize with simulation data"""
        self.simulation_data = simulation_data
        self.round_data = simulation_data.get('round_data', [])
        
        # Get total freelancer count for proper normalization
        freelancer_profiles = simulation_data.get('freelancer_profiles', {})
        self.total_freelancers = len(freelancer_profiles) if freelancer_profiles else \
                               simulation_data.get('simulation_config', {}).get('num_freelancers', 200)
        
    def plot_market_health_trends(self, save_path: str = "results/figures") -> None:
        """Plot market health trends over time"""
        if not self.round_data:
            logger.warning("No round data available for plotting")
            return
            
        # Extract health data
        rounds = []
        health_scores = []
        fill_rates = []
        
        for i, round_info in enumerate(self.round_data):
            round_num = round_info.get('round', 0)
            market_health = round_info.get('market_health', {})
            
            # Health score
            health_score = market_health.get('health_score', 0)
            
            # Fill rate (cumulative)
            cumulative_jobs_posted = sum(r.get('jobs_posted', 0) for r in self.round_data[:i+1])
            cumulative_jobs_filled = round_info.get('jobs_filled', 0)  # Total jobs filled so far
            fill_rate = cumulative_jobs_filled / cumulative_jobs_posted if cumulative_jobs_posted > 0 else 0
            
            rounds.append(round_num)
            health_scores.append(health_score)
            fill_rates.append(fill_rate)
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot health score
        ax1.plot(rounds, health_scores, 'b-', linewidth=2, label='Market Health Score')
        ax1.fill_between(rounds, health_scores, alpha=0.3, color='blue')
        ax1.set_ylabel('Health Score (0-1)')
        ax1.set_title('Market Health Trends Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add health grade zones
        ax1.axhline(y=0.75, color='green', linestyle='--', alpha=0.5, label='Excellent')
        ax1.axhline(y=0.60, color='yellow', linestyle='--', alpha=0.5, label='Good')
        ax1.axhline(y=0.40, color='orange', linestyle='--', alpha=0.5, label='Fair')
        ax1.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Poor')
        
        # Plot fill rate
        ax2.plot(rounds, fill_rates, 'g-', linewidth=2, label='Fill Rate')
        ax2.fill_between(rounds, fill_rates, alpha=0.3, color='green')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Fill Rate (0-1)')
        ax2.set_title('Job Fill Rate Trends')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / 'market_health_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Market health trends plot saved to {save_path / 'market_health_trends.png'}")
    
    def plot_competition_metrics(self, save_path: str = "results/figures") -> None:
        """Plot competition and bidding metrics over time"""
        if not self.round_data:
            logger.warning("No round data available for plotting")
            return
            
        # Extract competition data
        rounds = []
        avg_bids_per_job = []
        rejection_rates = []
        competitiveness_levels = []
        
        for round_info in self.round_data:
            round_num = round_info.get('round', 0)
            bid_dist = round_info.get('bid_distribution', {})
            bid_rejection = round_info.get('bid_rejection_metrics', {})
            market_health = round_info.get('market_health', {})
            
            rounds.append(round_num)
            avg_bids_per_job.append(bid_dist.get('avg_bids_per_job', 0))
            rejection_rates.append(bid_rejection.get('bid_rejection_rate', 0))
            competitiveness_levels.append(market_health.get('competitiveness_level', 'unknown'))
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot average bids per job
        ax1.plot(rounds, avg_bids_per_job, 'r-', linewidth=2, label='Avg Bids per Job')
        ax1.fill_between(rounds, avg_bids_per_job, alpha=0.3, color='red')
        ax1.set_ylabel('Average Bids per Job')
        ax1.set_title('Competition Intensity Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add optimal range
        ax1.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Optimal Min')
        ax1.axhline(y=4, color='green', linestyle='--', alpha=0.5, label='Optimal Max')
        ax1.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='High Competition')
        
        # Plot rejection rates
        ax2.plot(rounds, [r * 100 for r in rejection_rates], 'purple', linewidth=2, label='Rejection Rate')
        ax2.fill_between(rounds, [r * 100 for r in rejection_rates], alpha=0.3, color='purple')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Rejection Rate (%)')
        ax2.set_title('Bid Rejection Rate Trends')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / 'competition_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Competition trends plot saved to {save_path / 'competition_trends.png'}")
    
    def plot_outcome_diversity_trends(self, save_path: str = "results/figures") -> None:
        """Plot outcome diversity and work distribution trends"""
        if not self.round_data:
            logger.warning("No round data available for plotting")
            return
            
        # Extract diversity data
        rounds = []
        gini_coefficients = []
        freelancers_with_work = []
        participation_rates = []
        
        for round_info in self.round_data:
            round_num = round_info.get('round', 0)
            outcome_diversity = round_info.get('outcome_diversity', {})
            market_activity = round_info.get('market_activity', {})
            
            rounds.append(round_num)
            gini_coefficients.append(outcome_diversity.get('work_distribution_gini', 0))
            freelancers_with_work.append(outcome_diversity.get('freelancers_with_work', 0))
            participation_rates.append(market_activity.get('freelancer_participation_rate', 0))
        
        # Create subplot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot Gini coefficient
        ax1.plot(rounds, gini_coefficients, 'orange', linewidth=2, label='Gini Coefficient')
        ax1.fill_between(rounds, gini_coefficients, alpha=0.3, color='orange')
        ax1.set_ylabel('Gini Coefficient (0-1)')
        ax1.set_title('Work Distribution Inequality Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add inequality zones
        ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='High Equality')
        ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Low Equality')
        
        # Plot freelancers with work
        ax2.bar(rounds, freelancers_with_work, alpha=0.7, color='teal', label='Freelancers with Work')
        ax2.set_ylabel('Number of Freelancers')
        ax2.set_title('Freelancers with Active Work')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot participation rates
        ax3.plot(rounds, [r * 100 for r in participation_rates], 'brown', linewidth=2, label='Participation Rate')
        ax3.fill_between(rounds, [r * 100 for r in participation_rates], alpha=0.3, color='brown')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Participation Rate (%)')
        ax3.set_title('Freelancer Participation Rate')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / 'outcome_diversity_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Outcome diversity trends plot saved to {save_path / 'outcome_diversity_trends.png'}")
    
    def plot_engagement_trends(self, save_path: str = "results/figures") -> None:
        """Plot client and freelancer engagement trends"""
        if not self.round_data:
            logger.warning("No round data available for plotting")
            return
            
        # Extract engagement data
        rounds = []
        client_activity_rates = []
        freelancer_participation_rates = []
        fatigue_scores = []
        
        for round_info in self.round_data:
            round_num = round_info.get('round', 0)
            market_activity = round_info.get('market_activity', {})
            fatigue_indicators = round_info.get('fatigue_indicators', {})
            
            rounds.append(round_num)
            client_activity_rates.append(market_activity.get('client_activity_rate', 0))
            freelancer_participation_rates.append(market_activity.get('freelancer_participation_rate', 0))
            
            # Calculate average fatigue score
            # high_rejection_rate_freelancers is a raw count, normalize by total freelancers
            high_rejection_rate = fatigue_indicators.get('high_rejection_rate_freelancers', 0) / max(1, self.total_freelancers)
            fatigue_score = (
                high_rejection_rate +  # Now properly normalized (0-1)
                fatigue_indicators.get('declining_participation', 0) +  # Already 0-1
                fatigue_indicators.get('bid_selectivity_increase', 0)  # Already 0-1
            ) / 3.0
            fatigue_scores.append(fatigue_score)
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot activity rates
        ax1.plot(rounds, [r * 100 for r in client_activity_rates], 'navy', linewidth=2, label='Client Activity Rate')
        ax1.plot(rounds, [r * 100 for r in freelancer_participation_rates], 'darkgreen', linewidth=2, label='Freelancer Participation Rate')
        ax1.fill_between(rounds, [r * 100 for r in client_activity_rates], alpha=0.3, color='navy')
        ax1.fill_between(rounds, [r * 100 for r in freelancer_participation_rates], alpha=0.3, color='darkgreen')
        ax1.set_ylabel('Activity Rate (%)')
        ax1.set_title('Market Engagement Trends')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot fatigue scores
        ax2.plot(rounds, [f * 100 for f in fatigue_scores], 'red', linewidth=2, label='Fatigue Score')
        ax2.fill_between(rounds, [f * 100 for f in fatigue_scores], alpha=0.3, color='red')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Fatigue Score (%)')
        ax2.set_title('Freelancer Fatigue Trends')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / 'engagement_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Engagement trends plot saved to {save_path / 'engagement_trends.png'}")
    
    def plot_bidding_volume_trends(self, save_path: str = "results/figures") -> None:
        """Plot number of bids per round and jobs posted per round"""
        if not self.round_data:
            logger.warning("No round data available for plotting")
            return
            
        # Extract bidding volume data
        rounds = []
        total_bids = []
        jobs_posted = []
        jobs_filled = []
        
        for round_info in self.round_data:
            round_num = round_info.get('round', 0)
            
            rounds.append(round_num)
            total_bids.append(round_info.get('total_bids', 0))
            jobs_posted.append(round_info.get('jobs_posted', 0))
            jobs_filled.append(round_info.get('jobs_filled_this_round', 0))
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot bids per round and jobs posted/filled
        ax1.bar(rounds, total_bids, alpha=0.7, color='steelblue', label='Bids per Round', width=0.8)
        ax1.plot(rounds, jobs_posted, color='orange', linewidth=3, marker='o', label='Jobs Posted per Round', markersize=4)
        ax1.plot(rounds, jobs_filled, color='green', linewidth=3, marker='s', label='Jobs Filled per Round', markersize=4)
        
        ax1.set_ylabel('Number of Jobs/Bids')
        ax1.set_title('Bidding Volume and Job Activity Over Time')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot market efficiency: bids per job posted ratio
        bid_to_posted_ratio = [b/j if j > 0 else 0 for b, j in zip(total_bids, jobs_posted)]
        ax2.plot(rounds, bid_to_posted_ratio, 'purple', linewidth=3, label='Bids per Job Posted Ratio', marker='s', markersize=4)
        
        # Add optimal ratio reference lines
        ax2.axhline(y=2, color='green', linestyle='--', alpha=0.7, label='Healthy Min (2:1)')
        ax2.axhline(y=4, color='green', linestyle='--', alpha=0.7, label='Healthy Max (4:1)')
        ax2.axhline(y=1, color='orange', linestyle=':', alpha=0.7, label='Minimum Viable (1:1)')
        
        ax2.set_ylabel('Bids per Job Posted Ratio')
        ax2.set_xlabel('Round')
        ax2.set_title('Market Efficiency: Bids per Job Posted Over Time')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / 'bidding_volume_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Bidding volume trends plot saved to {save_path / 'bidding_volume_trends.png'}")
    
    def plot_budget_rate_adjustments(self, save_path: str = "results/figures") -> None:
        """Plot freelancer rate adjustments and client budget adjustments over time"""
        if not self.simulation_data.get('reflection_sessions'):
            logger.warning("No reflection sessions available for budget/rate plotting")
            return
            
        # Extract adjustment data from reflection sessions
        freelancer_adjustments = []
        client_budget_adjustments = []
        client_job_adjustments = []
        
        for session in self.simulation_data['reflection_sessions']:
            round_num = session.get('round', 0)
            agent_type = session.get('agent_type', '')
            agent_id = session.get('agent_id', '')
            
            if agent_type == 'freelancer' and 'rate_adjustment' in session:
                adj = session['rate_adjustment']
                freelancer_adjustments.append({
                    'round': round_num,
                    'agent_id': agent_id,
                    'old_rate': adj['old_rate'],
                    'new_rate': adj['new_rate'],
                    'change': adj['change'],
                    'reasoning': adj['reasoning']
                })
            
            elif agent_type == 'client':
                if 'budget_adjustment' in session:
                    adj = session['budget_adjustment']
                    client_budget_adjustments.append({
                        'round': round_num,
                        'agent_id': agent_id,
                        'old_multiplier': adj['old_multiplier'],
                        'new_multiplier': adj['new_multiplier'],
                        'change': adj['change']
                    })
                
                if 'job_budget_adjustments' in session:
                    for job_adj in session['job_budget_adjustments']:
                        client_job_adjustments.append({
                            'round': round_num,
                            'agent_id': agent_id,
                            'job_id': job_adj['job_id'],
                            'old_budget': job_adj['old_budget'],
                            'new_budget': job_adj['new_budget'],
                            'increase_percentage': job_adj['increase_percentage']
                        })
        
        if not freelancer_adjustments and not client_budget_adjustments and not client_job_adjustments:
            logger.warning("No budget or rate adjustments found to plot")
            return
        
        # Create subplot layout
        fig = plt.figure(figsize=(15, 12))
        
        # Plot 1: Freelancer Rate Evolution
        if freelancer_adjustments:
            ax1 = plt.subplot(3, 1, 1)
            
            # Group by freelancer for trend lines
            freelancer_rates = {}
            for adj in freelancer_adjustments:
                agent_id = adj['agent_id']
                if agent_id not in freelancer_rates:
                    freelancer_rates[agent_id] = {'rounds': [], 'rates': []}
                freelancer_rates[agent_id]['rounds'].append(adj['round'])
                freelancer_rates[agent_id]['rates'].append(adj['new_rate'])
            
            # Plot trend lines for each freelancer
            colors = plt.cm.tab10(np.linspace(0, 1, len(freelancer_rates)))
            for i, (agent_id, data) in enumerate(freelancer_rates.items()):
                ax1.plot(data['rounds'], data['rates'], 'o-', color=colors[i], 
                        linewidth=2, markersize=6, label=f'Freelancer {agent_id}')
            
            ax1.set_ylabel('Hourly Rate ($)')
            ax1.set_title('💰 Freelancer Rate Adjustments Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add adjustment arrows and annotations
            for adj in freelancer_adjustments:
                if adj['change'] == 'decrease':
                    ax1.annotate('⬇️', xy=(adj['round'], adj['new_rate']), 
                               xytext=(adj['round'], adj['new_rate'] + 5),
                               ha='center', fontsize=12, color='red')
                elif adj['change'] == 'increase':
                    ax1.annotate('⬆️', xy=(adj['round'], adj['new_rate']), 
                               xytext=(adj['round'], adj['new_rate'] - 5),
                               ha='center', fontsize=12, color='green')
        
        # Plot 2: Client Budget Multiplier Evolution
        if client_budget_adjustments:
            ax2 = plt.subplot(3, 1, 2)
            
            # Group by client for trend lines
            client_multipliers = {}
            for adj in client_budget_adjustments:
                agent_id = adj['agent_id']
                if agent_id not in client_multipliers:
                    client_multipliers[agent_id] = {'rounds': [], 'multipliers': []}
                client_multipliers[agent_id]['rounds'].append(adj['round'])
                client_multipliers[agent_id]['multipliers'].append(adj['new_multiplier'])
            
            # Plot trend lines for each client
            colors = plt.cm.tab20(np.linspace(0, 1, len(client_multipliers)))
            for i, (agent_id, data) in enumerate(client_multipliers.items()):
                ax2.plot(data['rounds'], data['multipliers'], 's-', color=colors[i], 
                        linewidth=2, markersize=6, label=f'Client {agent_id}')
            
            ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
            ax2.set_ylabel('Budget Multiplier')
            ax2.set_title('📊 Client Budget Strategy Adjustments Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Specific Job Budget Adjustments
        if client_job_adjustments:
            ax3 = plt.subplot(3, 1, 3)
            
            rounds = [adj['round'] for adj in client_job_adjustments]
            old_budgets = [adj['old_budget'] for adj in client_job_adjustments]
            new_budgets = [adj['new_budget'] for adj in client_job_adjustments]
            increases = [adj['increase_percentage'] for adj in client_job_adjustments]
            
            # Create bubble chart - size = increase percentage
            scatter = ax3.scatter(rounds, old_budgets, s=[inc*10 for inc in increases], 
                                c=increases, cmap='Reds', alpha=0.7, edgecolors='black')
            
            # Add arrows showing budget increases
            for adj in client_job_adjustments:
                ax3.annotate('', xy=(adj['round'], adj['new_budget']), 
                           xytext=(adj['round'], adj['old_budget']),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
                # Add percentage label
                ax3.text(adj['round'], adj['new_budget'] + 2, f"+{adj['increase_percentage']:.0f}%", 
                        ha='center', fontsize=9, color='green', weight='bold')
            
            ax3.set_xlabel('Round')
            ax3.set_ylabel('Job Budget ($)')
            ax3.set_title('🎯 Specific Job Budget Adjustments (Bubble size = % increase)')
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar for bubble colors
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Budget Increase %')
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / 'budget_rate_adjustments.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Budget/rate adjustments plot saved to {save_path / 'budget_rate_adjustments.png'}")
    
    def plot_agent_adaptation_timeline(self, save_path: str = "results/figures") -> None:
        """Plot a timeline showing how individual agents adapt over time"""
        if not self.simulation_data.get('reflection_sessions'):
            logger.warning("No reflection sessions available for adaptation timeline")
            return
        
        # Extract all agent adaptations by round
        agent_adaptations = {}
        
        for session in self.simulation_data['reflection_sessions']:
            round_num = session.get('round', 0)
            agent_type = session.get('agent_type', '')
            agent_id = session.get('agent_id', '')
            
            if agent_id not in agent_adaptations:
                agent_adaptations[agent_id] = {
                    'type': agent_type,
                    'rounds': [],
                    'actions': [],
                    'insights': []
                }
            
            # Record adaptations
            adaptations = agent_adaptations[agent_id]
            adaptations['rounds'].append(round_num)
            adaptations['insights'].append(session.get('key_insights', []))
            
            actions = []
            if agent_type == 'freelancer' and 'rate_adjustment' in session:
                adj = session['rate_adjustment']
                actions.append(f"Rate: ${adj['old_rate']}→${adj['new_rate']}")
            elif agent_type == 'client':
                if 'budget_adjustment' in session:
                    adj = session['budget_adjustment']
                    actions.append(f"Strategy: {adj['old_multiplier']:.2f}x→{adj['new_multiplier']:.2f}x")
                if 'job_budget_adjustments' in session:
                    job_count = len(session['job_budget_adjustments'])
                    actions.append(f"Jobs adjusted: {job_count}")
            
            adaptations['actions'].append(actions)
        
        if not agent_adaptations:
            logger.warning("No agent adaptations found to plot")
            return
        
        # Limit to top agents for readability: max 20 freelancers and 5 clients
        freelancers = {k: v for k, v in agent_adaptations.items() if v['type'] == 'freelancer'}
        clients = {k: v for k, v in agent_adaptations.items() if v['type'] == 'client'}
        
        # Sort by number of adaptations (rounds with actual changes)
        freelancers_sorted = sorted(freelancers.items(), 
                                  key=lambda x: len([r for r, a in zip(x[1]['rounds'], x[1]['actions']) if a]), 
                                  reverse=True)[:20]  # Top 20 most active freelancers
        
        clients_sorted = sorted(clients.items(), 
                              key=lambda x: len([r for r, a in zip(x[1]['rounds'], x[1]['actions']) if a]), 
                              reverse=True)[:5]   # Top 5 most active clients
        
        # Combine limited agents
        limited_adaptations = dict(freelancers_sorted + clients_sorted)
        
        # Create timeline plot
        fig, ax = plt.subplots(figsize=(15, 10))
        
        y_positions = {}
        y_counter = 0
        
        colors = {'freelancer': 'steelblue', 'client': 'darkorange'}
        
        for agent_id, data in limited_adaptations.items():
            y_positions[agent_id] = y_counter
            agent_type = data['type']
            
            # Plot timeline for this agent
            rounds = data['rounds']
            actions = data['actions']
            
            # Plot base timeline
            ax.plot([0, max(rounds) + 1], [y_counter, y_counter], 
                   color=colors[agent_type], alpha=0.3, linewidth=2)
            
            # Plot adaptation points
            for i, (round_num, action_list) in enumerate(zip(rounds, actions)):
                if action_list:  # Only plot if there were actual adjustments
                    ax.scatter(round_num, y_counter, s=100, color=colors[agent_type], 
                             edgecolors='black', linewidth=1, zorder=5)
                    
                    # Add action text
                    action_text = '\n'.join(action_list)
                    ax.annotate(action_text, xy=(round_num, y_counter), 
                              xytext=(round_num, y_counter + 0.3),
                              ha='center', va='bottom', fontsize=8,
                              bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor=colors[agent_type], alpha=0.7),
                              arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
            
            # Label agent
            ax.text(-0.5, y_counter, f"{agent_type.title()} {agent_id}", 
                   ha='right', va='center', fontweight='bold')
            
            y_counter += 1
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Agents')
        ax.set_title('🕒 Agent Adaptation Timeline - Top 20 Freelancers & Top 5 Clients')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Set limits and ticks
        if y_positions:
            ax.set_ylim(-0.5, max(y_positions.values()) + 0.5)
            ax.set_yticks([])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors['freelancer'], label='Freelancers'),
                          Patch(facecolor=colors['client'], label='Clients')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / 'agent_adaptation_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Agent adaptation timeline saved to {save_path / 'agent_adaptation_timeline.png'}")
    
    def plot_average_budget_changes_by_category(self, save_path: str = "results/figures") -> None:
        """Plot average budget changes over time aggregated by category for freelancers and jobs"""
        if not self.simulation_data.get('reflection_sessions'):
            logger.warning("No reflection sessions available for category budget plotting")
            return
        
        # Get freelancer categories from profiles
        freelancer_profiles = self.simulation_data.get('freelancer_profiles', {})
        freelancer_categories = {}
        for fid, profile in freelancer_profiles.items():
            freelancer_categories[fid] = profile.get('category', 'Unknown')
        
        # Get job categories from all_jobs
        all_jobs = self.simulation_data.get('all_jobs', [])
        job_categories = {}
        for job in all_jobs:
            job_categories[job.get('id', '')] = job.get('category', 'Unknown')
        
        # Extract adjustment data by category
        freelancer_rate_changes = {}  # category -> {round -> [changes]}
        job_budget_changes = {}       # category -> {round -> [changes]}
        
        for session in self.simulation_data['reflection_sessions']:
            round_num = session.get('round', 0)
            agent_type = session.get('agent_type', '')
            agent_id = session.get('agent_id', '')
            
            # Process freelancer rate adjustments
            if agent_type == 'freelancer' and 'rate_adjustment' in session:
                adj = session['rate_adjustment']
                category = freelancer_categories.get(agent_id, 'Unknown')
                
                old_rate = adj.get('old_rate', 0)
                new_rate = adj.get('new_rate', 0)
                
                if old_rate > 0:  # Only include valid rates
                    # Store absolute new rate instead of percentage change
                    if category not in freelancer_rate_changes:
                        freelancer_rate_changes[category] = {}
                    if round_num not in freelancer_rate_changes[category]:
                        freelancer_rate_changes[category][round_num] = []
                    
                    freelancer_rate_changes[category][round_num].append(new_rate)
            
            # Process job budget adjustments
            elif agent_type == 'client' and 'job_budget_adjustments' in session:
                for job_adj in session['job_budget_adjustments']:
                    job_id = job_adj.get('job_id', '')
                    category = job_categories.get(job_id, 'Unknown')
                    
                    old_budget = job_adj.get('old_budget', 0)
                    new_budget = job_adj.get('new_budget', 0)
                    
                    if old_budget > 0:  # Only include valid budgets
                        # Store absolute new budget instead of percentage change
                        if category not in job_budget_changes:
                            job_budget_changes[category] = {}
                        if round_num not in job_budget_changes[category]:
                            job_budget_changes[category][round_num] = []
                        
                        job_budget_changes[category][round_num].append(new_budget)
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Create consistent color mapping for all categories across both plots
        all_categories = set(freelancer_rate_changes.keys()) | set(job_budget_changes.keys())
        category_colors = {}
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_categories)))
        for i, category in enumerate(sorted(all_categories)):
            category_colors[category] = colors[i]
        
        # Plot 1: Freelancer Rate Changes by Category
        if freelancer_rate_changes:
            for category, rounds_data in freelancer_rate_changes.items():
                rounds = sorted(rounds_data.keys())
                avg_changes = []
                
                for round_num in rounds:
                    rates = rounds_data[round_num]
                    avg_rate = np.mean(rates) if rates else 0
                    avg_changes.append(avg_rate)
                
                # Count total adjustments across all rounds for this category
                total_adjustments = sum(len(rates) for rates in rounds_data.values())
                
                # Smooth the line for better visualization
                x_smooth, y_smooth = smooth_line(rounds, avg_changes)
                ax1.plot(x_smooth, y_smooth, '-', linewidth=3, color=category_colors[category],
                        label=f'{category} ({total_adjustments} adjustments)', alpha=0.8)
                # Add original points for reference
                ax1.plot(rounds, avg_changes, 'o', markersize=4, color=category_colors[category], alpha=0.6)
            
            ax1.set_ylabel('Average Hourly Rate ($)')
            ax1.set_title('💰 Average Freelancer Rates by Category Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax1.text(0.5, 0.5, 'No freelancer rate changes found', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('💰 Average Freelancer Rates by Category Over Time')
        
        # Plot 2: Job Budget Changes by Category
        if job_budget_changes:
            for category, rounds_data in job_budget_changes.items():
                rounds = sorted(rounds_data.keys())
                avg_changes = []
                
                for round_num in rounds:
                    budgets = rounds_data[round_num]
                    avg_budget = np.mean(budgets) if budgets else 0
                    avg_changes.append(avg_budget)
                
                # Count total adjustments across all rounds for this category
                total_adjustments = sum(len(budgets) for budgets in rounds_data.values())
                
                # Smooth the line for better visualization
                x_smooth, y_smooth = smooth_line(rounds, avg_changes)
                ax2.plot(x_smooth, y_smooth, '-', linewidth=3, color=category_colors[category],
                        label=f'{category} ({total_adjustments} adjustments)', alpha=0.8)
                # Add original points for reference
                ax2.plot(rounds, avg_changes, 's', markersize=4, color=category_colors[category], alpha=0.6)
            
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Average Job Budget ($)')
            ax2.set_title('📊 Average Job Budgets by Category Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax2.text(0.5, 0.5, 'No job budget changes found', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_xlabel('Round')
            ax2.set_title('📊 Average Job Budgets by Category Over Time')
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / 'category_budget_changes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Category budget changes plot saved to {save_path / 'category_budget_changes.png'}")
        
        # Log summary statistics
        if freelancer_rate_changes:
            logger.info("Freelancer rate summary:")
            for category, rounds_data in freelancer_rate_changes.items():
                all_rates = [rate for rates in rounds_data.values() for rate in rates]
                avg_rate = np.mean(all_rates) if all_rates else 0
                logger.info(f"  {category}: {len(all_rates)} rate adjustments, avg ${avg_rate:.1f}/hr")
        
        if job_budget_changes:
            logger.info("Job budget summary:")
            for category, rounds_data in job_budget_changes.items():
                all_budgets = [budget for budgets in rounds_data.values() for budget in budgets]
                avg_budget = np.mean(all_budgets) if all_budgets else 0
                logger.info(f"  {category}: {len(all_budgets)} budget adjustments, avg ${avg_budget:.0f}")

    def generate_all_trend_plots(self, save_path: str = "results/figures") -> None:
        """Generate all trend plots"""
        logger.info("Generating market trend plots...")
        
        self.plot_market_health_trends(save_path)
        self.plot_competition_metrics(save_path)
        self.plot_outcome_diversity_trends(save_path)
        self.plot_engagement_trends(save_path)
        self.plot_bidding_volume_trends(save_path)
        self.plot_budget_rate_adjustments(save_path)  # 🆕 Budget/rate evolution
        self.plot_agent_adaptation_timeline(save_path)  # 🆕 Adaptation timeline
        self.plot_average_budget_changes_by_category(save_path)  # 🆕 Category budget changes
        
        logger.info("All market trend plots generated successfully")

def plot_market_trends_from_file(simulation_file: str, save_path: str = "results/figures") -> None:
    """Generate trend plots from simulation file"""
    with open(simulation_file, 'r') as f:
        simulation_data = json.load(f)
    
    plotter = MarketTrendPlotter(simulation_data)
    plotter.generate_all_trend_plots(save_path)
