"""
Analysis Module for Marketplace Simulation

This module serves as the main entrypoint for all marketplace analysis functionality.
It coordinates between different analysis modules and provides a clean API.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from .core.market_metrics import MarketplaceAnalyzer
from ..marketplace.market_metrics_utils import (
    calculate_trend_direction,
    calculate_freelancer_tier,
    calculate_client_tier
)

logger = logging.getLogger(__name__)

class MarketplaceAnalysis:
    """Main analysis coordinator for marketplace simulation"""
    
    def __init__(self, simulation_file: str):
        """Initialize analysis with simulation data"""
        self.simulation_file = Path(simulation_file)
        if not self.simulation_file.exists():
            raise FileNotFoundError(f"Simulation file not found: {simulation_file}")
        
        # Initialize analyzer
        self.market_analyzer = MarketplaceAnalyzer(simulation_file)
        
        # Load data
        self.market_analyzer.load_data()
        
        # Load simulation data for reputation analysis
        with open(self.simulation_file) as f:
            self.simulation_data = json.load(f)
    
    def analyze_market_metrics(self) -> Dict[str, Any]:
        """Run market efficiency and performance analysis"""
        logger.info("Analyzing market metrics...")
        
        metrics = {
            'market_efficiency': self.market_analyzer.generate_market_efficiency_analysis(),
            'skill_distribution': self.market_analyzer.analyze_skill_distribution(),
            'job_duration': self.market_analyzer.analyze_job_duration_impact(),
            'job_complexity': self.market_analyzer.analyze_job_complexity(),
            'unfilled_jobs': self.market_analyzer.analyze_unfilled_jobs(),
            'saturation_analysis': self.analyze_market_saturation()
        }
        
        return metrics
    
    def analyze_market_saturation(self) -> Dict[str, Any]:
        """Analyze market saturation and health metrics"""
        logger.info("Analyzing market saturation...")
        
        saturation_analysis = {
            'job_fill_rate_trends': self._analyze_fill_rate_trends(),
            'bid_volume_analysis': self._analyze_bid_volumes(),
            'bid_rejection_analysis': self._analyze_bid_rejection_trends(),
            'freelancer_outcome_diversity': self._analyze_outcome_diversity(),
            'client_activity_patterns': self._analyze_client_activity_patterns(),
            'freelancer_fatigue_indicators': self._analyze_freelancer_fatigue(),
            'emerging_strategies': self._analyze_strategy_evolution(),
            'bias_indicators': self._analyze_bias_patterns(),
            'market_health_score': self._calculate_market_health_score()
        }
        
        return saturation_analysis
    
    def analyze_bidding_behavior(self) -> Dict[str, Any]:
        """Analyze bidding patterns and strategies"""
        logger.info("Analyzing bidding behavior...")
        
        behavior = {
            'bidding_strategies': self.market_analyzer.analyze_bidding_strategies(),
            'decision_patterns': self._analyze_decision_patterns()
        }
        
        return behavior
    
    def analyze_agent_learning(self) -> Dict[str, Any]:
        """Analyze agent learning and adaptation"""
        logger.info("Analyzing agent learning...")
        
        learning = {
            'economic_patterns': self.market_analyzer.analyze_economic_patterns()
        }
        
        return learning
    
    def generate_visualizations(self, output_dir: str = "results/figures"):
        """Generate all analysis visualizations"""
        logger.info("Generating visualizations...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate market analysis plots
        self.market_analyzer.generate_visualizations(output_dir)
    
    def generate_reports(self, output_dir: str = "results") -> Dict[str, str]:
        """Generate analysis reports"""
        logger.info("Generating analysis reports...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate reports
        reports = {}
        
        # Market analysis report
        market_report = self.market_analyzer.generate_summary_report()
        market_report_path = output_path / "analysis_results.json"
        with open(market_report_path, 'w') as f:
            import json
            json.dump(market_report, f, indent=2, default=str)
        reports['market_analysis'] = str(market_report_path)
        
        return reports
    
    def run_full_analysis(self, output_dir: str = "results") -> Dict[str, Any]:
        """Run complete analysis pipeline"""
        logger.info("Running full analysis pipeline...")
        
        # Run all analyses
        results = {
            'market_metrics': self.analyze_market_metrics(),
            'bidding_behavior': self.analyze_bidding_behavior(),
            'agent_learning': self.analyze_agent_learning(),
            'reputation_analysis': self.analyze_reputation_dynamics()
        }
        
        # Generate visualizations
        self.generate_visualizations(f"{output_dir}/figures")
        
        # Generate reports
        reports = self.generate_reports(output_dir)
        results['reports'] = reports
        
        logger.info("Analysis pipeline completed!")
        return results
    
    # Saturation Analysis Methods
    def _analyze_fill_rate_trends(self) -> Dict[str, Any]:
        """Analyze job fill rate trends over time"""
        try:
            round_data = self.market_analyzer.round_data
            if round_data is None or round_data.empty:
                return {'error': 'No round data available'}
            
            # Calculate fill rates by round
            fill_rates = []
            for _, round_row in round_data.iterrows():
                jobs_posted = round_row.get('jobs_posted', 0)
                jobs_filled = round_row.get('jobs_filled_this_round', round_row.get('jobs_filled', 0))
                fill_rate = jobs_filled / jobs_posted if jobs_posted > 0 else 0
                fill_rates.append({
                    'round': round_row.get('round', 0),
                    'fill_rate': fill_rate,
                    'jobs_posted': jobs_posted,
                    'jobs_filled': jobs_filled
                })
            
            # Calculate trend metrics
            recent_rates = [r['fill_rate'] for r in fill_rates[-3:]] if len(fill_rates) >= 3 else [r['fill_rate'] for r in fill_rates]
            trend = 'improving' if len(recent_rates) >= 2 and recent_rates[-1] > recent_rates[0] else 'declining' if len(recent_rates) >= 2 and recent_rates[-1] < recent_rates[0] else 'stable'
            
            return {
                'fill_rate_by_round': fill_rates,
                'overall_trend': trend,
                'average_fill_rate': sum(r['fill_rate'] for r in fill_rates) / len(fill_rates) if fill_rates else 0,
                'latest_fill_rate': fill_rates[-1]['fill_rate'] if fill_rates else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing fill rate trends: {e}")
            return {'error': str(e)}
    
    def _analyze_bid_volumes(self) -> Dict[str, Any]:
        """Analyze bid volume patterns and saturation indicators with historical trends"""
        try:
            round_data = self.market_analyzer.round_data
            if round_data is None or round_data.empty:
                return {'error': 'No round data available'}
            
            bid_volume_analysis = {}
            historical_health = []
            historical_competitiveness = []
            
            # Extract enhanced bid distribution data from round_data
            for _, round_row in round_data.iterrows():
                round_num = round_row.get('round', 0)
                bid_dist = round_row.get('bid_distribution', {})
                market_health = round_row.get('market_health', {})
                
                if isinstance(bid_dist, dict):
                    avg_bids = bid_dist.get('avg_bids_per_job', 0)
                    bid_volume_analysis[round_num] = {
                        'avg_bids_per_job': avg_bids,
                        'max_bids_per_job': bid_dist.get('max_bids_per_job', 0),
                        'jobs_with_no_bids': bid_dist.get('jobs_with_no_bids', 0),
                        'jobs_with_excess_bids': bid_dist.get('jobs_with_excess_bids', 0),
                        'competitiveness_level': market_health.get('competitiveness_level', 'unknown'),
                        'health_score': market_health.get('health_score', 0),
                        'health_grade': market_health.get('health_grade', 'unknown'),
                        'saturation_risk': market_health.get('market_saturation_risk', 0)
                    }
                    
                    # Track historical trends
                    historical_health.append(market_health.get('health_score', 0))
                    historical_competitiveness.append(avg_bids)
            
            # Calculate trend metrics
            avg_bids_trend = [data['avg_bids_per_job'] for data in bid_volume_analysis.values()]
            saturation_risk = 'high' if any(avg > 8 for avg in avg_bids_trend) else 'moderate' if any(avg > 5 for avg in avg_bids_trend) else 'low'
            
            # Calculate trend directions
            health_trend = calculate_trend_direction(historical_health)
            competition_trend = calculate_trend_direction(historical_competitiveness)
            
            return {
                'bid_volume_by_round': bid_volume_analysis,
                'saturation_risk_level': saturation_risk,
                'average_bids_per_job': sum(avg_bids_trend) / len(avg_bids_trend) if avg_bids_trend else 0,
                'peak_bid_volume': max(avg_bids_trend) if avg_bids_trend else 0,
                'health_trend': health_trend,
                'competition_trend': competition_trend,
                'current_health_grade': bid_volume_analysis[max(bid_volume_analysis.keys())]['health_grade'] if bid_volume_analysis else 'unknown',
                'current_competitiveness': bid_volume_analysis[max(bid_volume_analysis.keys())]['competitiveness_level'] if bid_volume_analysis else 'unknown'
            }
        except Exception as e:
            logger.error(f"Error analyzing bid volumes: {e}")
            return {'error': str(e)}
    

    
    def _analyze_bid_rejection_trends(self) -> Dict[str, Any]:
        """Analyze bid rejection rates and trends over time"""
        try:
            round_data = self.market_analyzer.round_data
            if round_data is None or round_data.empty:
                return {'error': 'No round data available'}
            
            rejection_analysis = {}
            rejection_rates = []
            
            # Extract bid rejection data from round_data
            for _, round_row in round_data.iterrows():
                round_num = round_row.get('round', 0)
                rejection_metrics = round_row.get('bid_rejection_metrics', {})
                
                if isinstance(rejection_metrics, dict):
                    rejection_rate = rejection_metrics.get('bid_rejection_rate', 0)
                    success_rate = rejection_metrics.get('bid_success_rate', 0)
                    total_bids = rejection_metrics.get('total_bids_this_round', 0)
                    rejected_bids = rejection_metrics.get('rejected_bids_this_round', 0)
                    
                    rejection_rates.append(rejection_rate)
                    rejection_analysis[round_num] = {
                        'rejection_rate': rejection_rate,
                        'success_rate': success_rate,
                        'total_bids': total_bids,
                        'rejected_bids': rejected_bids
                    }
            
            # Calculate trend analysis
            if len(rejection_rates) >= 3:
                early_avg = sum(rejection_rates[:3]) / 3
                late_avg = sum(rejection_rates[-3:]) / 3
                trend_direction = 'increasing' if late_avg > early_avg + 0.05 else 'decreasing' if late_avg < early_avg - 0.05 else 'stable'
            else:
                early_avg = late_avg = sum(rejection_rates) / len(rejection_rates) if rejection_rates else 0
                trend_direction = 'stable'
            
            avg_rejection_rate = sum(rejection_rates) / len(rejection_rates) if rejection_rates else 0
            
            # Assess market competitiveness based on rejection rates
            if avg_rejection_rate > 0.8:
                competitiveness = 'extremely_high'
            elif avg_rejection_rate > 0.6:
                competitiveness = 'high'
            elif avg_rejection_rate > 0.4:
                competitiveness = 'moderate'
            elif avg_rejection_rate > 0.2:
                competitiveness = 'low'
            else:
                competitiveness = 'very_low'
            
            return {
                'rejection_by_round': rejection_analysis,
                'average_rejection_rate': avg_rejection_rate,
                'trend_direction': trend_direction,
                'early_period_avg': early_avg,
                'late_period_avg': late_avg,
                'competitiveness_level': competitiveness,
                'peak_rejection_rate': max(rejection_rates) if rejection_rates else 0,
                'lowest_rejection_rate': min(rejection_rates) if rejection_rates else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing bid rejection trends: {e}")
            return {'error': str(e)}
    
    def _analyze_outcome_diversity(self) -> Dict[str, Any]:
        """Analyze diversity in market outcomes with historical Gini coefficient tracking"""
        try:
            round_data = self.market_analyzer.round_data
            if round_data is None or round_data.empty:
                return {'error': 'No round data available'}
            
            historical_gini = []
            historical_freelancers_with_work = []
            historical_participation = []
            
            # Extract outcome diversity data from round_data
            for _, round_row in round_data.iterrows():
                outcome_diversity = round_row.get('outcome_diversity', {})
                if isinstance(outcome_diversity, dict):
                    gini = outcome_diversity.get('work_distribution_gini', 0)
                    freelancers_with_work = outcome_diversity.get('freelancers_with_work', 0)
                    total_freelancers = outcome_diversity.get('total_freelancers', 1)
                    participation = outcome_diversity.get('participation_rate', 0)
                    
                    historical_gini.append(gini)
                    historical_freelancers_with_work.append(freelancers_with_work)
                    historical_participation.append(participation)
            
            # Calculate current and trend metrics
            current_gini = historical_gini[-1] if historical_gini else 0
            current_freelancers_with_work = historical_freelancers_with_work[-1] if historical_freelancers_with_work else 0
            total_freelancers = round_data.iloc[-1].get('outcome_diversity', {}).get('total_freelancers', 1) if not round_data.empty else 1
            
            # Determine diversity level
            if current_gini < 0.3:
                diversity_level = 'high'  # More equal distribution
            elif current_gini < 0.6:
                diversity_level = 'moderate'
            else:
                diversity_level = 'low'  # Very unequal distribution
            
            # Calculate trends
            gini_trend = calculate_trend_direction(historical_gini)
            participation_trend = calculate_trend_direction(historical_participation)
            
            return {
                'current_gini_coefficient': current_gini,
                'diversity_level': diversity_level,
                'freelancers_with_work': current_freelancers_with_work,
                'total_freelancers': total_freelancers,
                'work_distribution_ratio': current_freelancers_with_work / total_freelancers if total_freelancers > 0 else 0,
                'gini_trend': gini_trend,
                'participation_trend': participation_trend,
                'historical_gini': historical_gini,
                'historical_freelancers_with_work': historical_freelancers_with_work,
                'average_gini': sum(historical_gini) / len(historical_gini) if historical_gini else 0,
                'peak_inequality': max(historical_gini) if historical_gini else 0,
                'most_equal_round': min(historical_gini) if historical_gini else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing outcome diversity: {e}")
            return {'error': str(e)}
    
    def _analyze_client_activity_patterns(self) -> Dict[str, Any]:
        """Analyze client posting patterns and engagement levels"""
        try:
            round_data = self.market_analyzer.round_data
            if round_data is None or round_data.empty:
                return {'error': 'No round data available'}
            
            # Extract client activity data
            client_activity = {}
            for _, round_row in round_data.iterrows():
                round_num = round_row.get('round', 0)
                market_activity = round_row.get('market_activity', {})
                if isinstance(market_activity, dict):
                    client_activity[round_num] = {
                        'active_clients': market_activity.get('active_clients', 0),
                        'clients_on_cooldown': market_activity.get('clients_on_cooldown', 0),
                        'client_activity_rate': market_activity.get('client_activity_rate', 0),
                        'posting_frequency': market_activity.get('client_posting_frequency', {})
                    }
            
            # Calculate engagement trends
            activity_rates = [data['client_activity_rate'] for data in client_activity.values()]
            avg_activity_rate = sum(activity_rates) / len(activity_rates) if activity_rates else 0
            
            # Determine if client engagement is declining
            if len(activity_rates) >= 3:
                recent_trend = activity_rates[-1] - activity_rates[-3]
                engagement_trend = 'increasing' if recent_trend > 0.1 else 'decreasing' if recent_trend < -0.1 else 'stable'
            else:
                engagement_trend = 'stable'
            
            engagement_assessment = 'high' if avg_activity_rate > 0.6 else 'moderate' if avg_activity_rate > 0.3 else 'low'
            
            return {
                'client_activity_by_round': client_activity,
                'average_activity_rate': avg_activity_rate,
                'engagement_trend': engagement_trend,
                'engagement_assessment': engagement_assessment,
                'note': 'Clients have cooldown periods but do not permanently leave'
            }
        except Exception as e:
            logger.error(f"Error analyzing client activity patterns: {e}")
            return {'error': str(e)}
    
    def _analyze_freelancer_fatigue(self) -> Dict[str, Any]:
        """Analyze freelancer fatigue indicators and sentiment trends"""
        try:
            round_data = self.market_analyzer.round_data
            if round_data is None or round_data.empty:
                return {'error': 'No round data available'}
            
            fatigue_analysis = {}
            overall_fatigue_trend = []
            
            for _, round_row in round_data.iterrows():
                round_num = round_row.get('round', 0)
                fatigue_indicators = round_row.get('fatigue_indicators', {})
                
                if isinstance(fatigue_indicators, dict):
                    high_rejection_freelancers = fatigue_indicators.get('high_rejection_rate_freelancers', 0)
                    declining_participation = fatigue_indicators.get('declining_participation', 0)
                    bid_selectivity_increase = fatigue_indicators.get('bid_selectivity_increase', 0)
                    
                    # Get total freelancer count for proper normalization
                    total_freelancers = len(self.simulation_data.get('freelancer_profiles', {}))
                    if total_freelancers == 0:
                        total_freelancers = self.simulation_data.get('simulation_config', {}).get('num_freelancers', 200)
                    
                    # Normalize high rejection rate freelancers count by total freelancers
                    high_rejection_rate = high_rejection_freelancers / max(1, total_freelancers)
                    fatigue_score = (high_rejection_rate + declining_participation + bid_selectivity_increase) / 3.0
                    overall_fatigue_trend.append(fatigue_score)
                    
                    fatigue_analysis[round_num] = {
                        'high_rejection_freelancers': high_rejection_freelancers,
                        'declining_participation': declining_participation,
                        'bid_selectivity_increase': bid_selectivity_increase,
                        'fatigue_score': fatigue_score
                    }
            
            # Assess overall fatigue trend
            avg_fatigue = sum(overall_fatigue_trend) / len(overall_fatigue_trend) if overall_fatigue_trend else 0
            fatigue_level = 'high' if avg_fatigue > 0.6 else 'moderate' if avg_fatigue > 0.3 else 'low'
            
            return {
                'fatigue_by_round': fatigue_analysis,
                'overall_fatigue_level': fatigue_level,
                'average_fatigue_score': avg_fatigue,
                'fatigue_trend': 'increasing' if len(overall_fatigue_trend) >= 2 and overall_fatigue_trend[-1] > overall_fatigue_trend[0] else 'stable'
            }
        except Exception as e:
            logger.error(f"Error analyzing freelancer fatigue: {e}")
            return {'error': str(e)}
    
    def _analyze_strategy_evolution(self) -> Dict[str, Any]:
        """Analyze emerging strategies and behavioral adaptations"""
        try:
            # This would require analyzing bidding decision patterns over time
            # For now, return a placeholder implementation
            return {
                'strategy_clusters': {},
                'adaptation_patterns': {},
                'emerging_behaviors': [],
                'note': 'Strategy evolution analysis requires more detailed decision tracking - placeholder implementation'
            }
        except Exception as e:
            logger.error(f"Error analyzing strategy evolution: {e}")
            return {'error': str(e)}
    
    def _analyze_bias_patterns(self) -> Dict[str, Any]:
        """Analyze potential bias patterns in hiring outcomes"""
        try:
            freelancer_data = self.market_analyzer.freelancer_states
            if freelancer_data is None or freelancer_data.empty:
                return {'error': 'No freelancer data available'}
            
            # Location bias analysis removed - geographic information excluded to prevent bias
            
            return {
                'note': 'Geographic bias analysis not applicable - location information excluded from framework to prevent demographic bias'
            }
        except Exception as e:
            logger.error(f"Error analyzing bias patterns: {e}")
            return {'error': str(e)}
    
    def _calculate_market_health_score(self) -> Dict[str, Any]:
        """Calculate overall market health score using historical data"""
        try:
            round_data = self.market_analyzer.round_data
            if round_data is None or round_data.empty:
                return {'error': 'No round data available'}
            
            # Extract health scores directly from simulation data (which now calculates them)
            health_scores = []
            health_grades = []
            
            for _, round_row in round_data.iterrows():
                market_health = round_row.get('market_health', {})
                if isinstance(market_health, dict):
                    health_score = market_health.get('health_score', 0)
                    health_grade = market_health.get('health_grade', 'unknown')
                    health_scores.append(health_score)
                    health_grades.append(health_grade)
            
            if not health_scores:
                # No health scores found in simulation data
                return {'error': 'No health scores found in simulation data'}
            
            # Calculate aggregated metrics
            current_health_score = health_scores[-1] if health_scores else 0
            current_health_grade = health_grades[-1] if health_grades else 'unknown'
            average_health_score = sum(health_scores) / len(health_scores) if health_scores else 0
            health_trend = calculate_trend_direction(health_scores)
            
            # Count grade distribution
            grade_distribution = {}
            for grade in health_grades:
                grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
            
            return {
                'overall_health_score': current_health_score,
                'health_grade': current_health_grade,
                'average_health_score': average_health_score,
                'health_trend': health_trend,
                'historical_scores': health_scores,
                'grade_distribution': grade_distribution,
                'peak_health': max(health_scores) if health_scores else 0,
                'lowest_health': min(health_scores) if health_scores else 0,
                'health_volatility': max(health_scores) - min(health_scores) if len(health_scores) > 1 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating market health score: {e}")
            return {'error': str(e)}
    

    
    def _analyze_decision_patterns(self) -> Dict[str, Any]:
        """Analyze freelancer decision patterns and trends"""
        try:
            # Load all_decisions from the simulation data
            simulation_file = Path(self.simulation_file)
            if not simulation_file.exists():
                return {'error': 'Simulation file not found'}
            
            with open(simulation_file) as f:
                simulation_data = json.load(f)
            
            all_decisions = simulation_data.get('all_decisions', [])
            
            if not all_decisions:
                return {
                    'total_decisions': 0,
                    'bid_rate': 0.0,
                    'decision_trends': {},
                    'bid_rate_by_round': {},
                    'reasoning_analysis': {},
                    'note': 'No decision data available - decisions may not have been recorded'
                }
            
            # Calculate basic metrics
            total_decisions = len(all_decisions)
            yes_decisions = len([d for d in all_decisions if d.get('decision') == 'yes'])
            bid_rate = yes_decisions / total_decisions if total_decisions > 0 else 0.0
            
            # Analyze decision trends by round
            round_decisions = {}
            round_bid_rates = {}
            
            for decision in all_decisions:
                round_num = decision.get('round', 0)
                if round_num not in round_decisions:
                    round_decisions[round_num] = {'total': 0, 'yes': 0}
                
                round_decisions[round_num]['total'] += 1
                if decision.get('decision') == 'yes':
                    round_decisions[round_num]['yes'] += 1
            
            # Calculate bid rate by round
            for round_num, counts in round_decisions.items():
                round_bid_rates[round_num] = counts['yes'] / counts['total'] if counts['total'] > 0 else 0.0
            
            # Analyze reasoning patterns (basic keyword analysis)
            reasoning_keywords = {}
            for decision in all_decisions:
                reasoning = decision.get('reasoning', '').lower()
                if 'budget' in reasoning or 'rate' in reasoning:
                    reasoning_keywords['budget_concerns'] = reasoning_keywords.get('budget_concerns', 0) + 1
                if 'skill' in reasoning or 'expertise' in reasoning:
                    reasoning_keywords['skill_mismatch'] = reasoning_keywords.get('skill_mismatch', 0) + 1
                if 'experience' in reasoning:
                    reasoning_keywords['experience_related'] = reasoning_keywords.get('experience_related', 0) + 1
                if 'align' in reasoning or 'match' in reasoning:
                    reasoning_keywords['alignment_issues'] = reasoning_keywords.get('alignment_issues', 0) + 1
            
            return {
                'total_decisions': total_decisions,
                'bid_rate': bid_rate,
                'yes_decisions': yes_decisions,
                'no_decisions': total_decisions - yes_decisions,
                'decision_trends': round_decisions,
                'bid_rate_by_round': round_bid_rates,
                'reasoning_analysis': reasoning_keywords,
                'avg_decisions_per_round': total_decisions / len(round_decisions) if round_decisions else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing decision patterns: {e}")
            return {'error': str(e)}
    
    def analyze_reputation_dynamics(self) -> Dict:
        """Analyze reputation system dynamics and impact on marketplace behavior"""
        try:
            # Load historical reputation data from round_data
            round_data = self.simulation_data.get('round_data', [])
            if not round_data:
                return {'error': 'No round data available for reputation analysis'}
            
            # Extract reputation progression over time
            reputation_history = self._extract_reputation_history(round_data)
            if not reputation_history:
                return {'error': 'No reputation snapshots found in round data'}
            
            # Analyze reputation progression patterns
            freelancer_progression = self._analyze_reputation_progression(reputation_history, 'freelancers')
            client_progression = self._analyze_reputation_progression(reputation_history, 'clients')
            
            # Analyze final state for comparison
            final_round = round_data[-1]
            final_reputation = final_round.get('reputation_snapshot', {})
            
            freelancer_final_data = [rep for rep in final_reputation.get('freelancers', {}).values() if rep is not None]
            client_final_data = [rep for rep in final_reputation.get('clients', {}).values() if rep is not None]
            
            # Analyze current reputation distribution
            freelancer_reputation = self._analyze_freelancer_reputation_distribution(freelancer_final_data)
            client_reputation = self._analyze_client_reputation_distribution(client_final_data)
            
            # Analyze reputation impact on marketplace outcomes
            reputation_impact = self._analyze_reputation_impact(freelancer_final_data, client_final_data)
            
            return {
                'reputation_progression': {
                    'freelancers': freelancer_progression,
                    'clients': client_progression
                },
                'current_reputation_distribution': {
                    'freelancers': freelancer_reputation,
                    'clients': client_reputation
                },
                'reputation_impact': reputation_impact,
                'historical_data_points': len(reputation_history),
                'total_freelancers_analyzed': len(freelancer_final_data),
                'total_clients_analyzed': len(client_final_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing reputation dynamics: {e}")
            return {'error': str(e)}
    
    def _analyze_freelancer_reputation_distribution(self, freelancer_data: List[Dict]) -> Dict:
        """Analyze freelancer reputation tier distribution and metrics"""
        if not freelancer_data:
            return {}
        
        # Track tier distribution
        tier_counts = {'New': 0, 'Established': 0, 'Expert': 0, 'Elite': 0}
        completion_rates = []
        earnings_by_tier = {'New': [], 'Established': [], 'Expert': [], 'Elite': []}
        total_jobs_hired = 0
        total_successful_jobs = 0
        
        for freelancer in freelancer_data:
            # Use shared tier calculation function
            hired = freelancer.get('total_hired', 0)
            completed = freelancer.get('completed_jobs', 0)
            
            completion_rate = completed / hired if hired > 0 else 0.0
            completion_rates.append(completion_rate)
            
            # Use shared tier calculation from market_metrics_utils
            tier = calculate_freelancer_tier(freelancer)
            
            tier_counts[tier] += 1
            
            # Track earnings by tier (estimate based on jobs completed)
            estimated_earnings = completed * 50  # Rough estimate
            earnings_by_tier[tier].append(estimated_earnings)
            
            total_jobs_hired += hired
            total_successful_jobs += completed
        
        # Calculate average earnings by tier
        avg_earnings_by_tier = {}
        top_earning_tier = 'New'
        max_avg_earnings = 0
        
        for tier, earnings_list in earnings_by_tier.items():
            if earnings_list:
                avg_earnings = sum(earnings_list) / len(earnings_list)
                avg_earnings_by_tier[tier] = avg_earnings
                if avg_earnings > max_avg_earnings:
                    max_avg_earnings = avg_earnings
                    top_earning_tier = tier
        
        return {
            'tier_distribution': tier_counts,
            'average_completion_rate': sum(completion_rates) / len(completion_rates) if completion_rates else 0,
            'completion_rates_by_tier': {
                tier: sum(1 for i, f in enumerate(freelancer_data) 
                         if calculate_freelancer_tier(f) == tier 
                         and completion_rates[i] > 0.8) / max(1, tier_counts[tier])
                for tier in tier_counts.keys()
            },
            'avg_earnings_by_tier': avg_earnings_by_tier,
            'top_earning_tier': top_earning_tier,
            'total_jobs_hired': total_jobs_hired,
            'total_successful_jobs': total_successful_jobs,
            'overall_success_rate': total_successful_jobs / total_jobs_hired if total_jobs_hired > 0 else 0
        }
    
    def _analyze_client_reputation_distribution(self, client_data: List[Dict]) -> Dict:
        """Analyze client reputation tier distribution and metrics"""
        if not client_data:
            return {}
        
        tier_counts = {'New': 0, 'Established': 0, 'Expert': 0, 'Elite': 0}
        hire_success_rates = []
        total_jobs_posted = 0
        total_successful_hires = 0
        
        for client in client_data:
            jobs_posted = client.get('jobs_posted', 0)
            successful_hires = client.get('successful_hires', 0)
            
            hire_success_rate = successful_hires / jobs_posted if jobs_posted > 0 else 0.0
            hire_success_rates.append(hire_success_rate)
            
            # Use shared tier calculation from market_metrics_utils
            tier = calculate_client_tier(client)
            
            tier_counts[tier] += 1
            total_jobs_posted += jobs_posted
            total_successful_hires += successful_hires
        
        return {
            'tier_distribution': tier_counts,
            'average_hire_success_rate': sum(hire_success_rates) / len(hire_success_rates) if hire_success_rates else 0,
            'total_jobs_posted': total_jobs_posted,
            'total_successful_hires': total_successful_hires,
            'overall_hire_success_rate': total_successful_hires / total_jobs_posted if total_jobs_posted > 0 else 0
        }
    
    def _analyze_reputation_impact(self, freelancer_data: List[Dict], client_data: List[Dict]) -> Dict:
        """Analyze the impact of reputation on marketplace outcomes"""
        if not freelancer_data:
            return {}
        
        # Calculate hiring rates by freelancer tier
        hiring_rates_by_tier = {}
        tier_job_counts = {'New': 0, 'Established': 0, 'Expert': 0, 'Elite': 0}
        
        for freelancer in freelancer_data:
            tier = calculate_freelancer_tier(freelancer)
            hired = freelancer.get('total_hired', 0)
            bids = freelancer.get('total_bids', 0)
            
            if tier not in hiring_rates_by_tier:
                hiring_rates_by_tier[tier] = []
            
            hire_rate = hired / bids if bids > 0 else 0
            hiring_rates_by_tier[tier].append(hire_rate)
            tier_job_counts[tier] += hired
        
        # Calculate average hiring rates by tier
        avg_hiring_rates = {}
        for tier, rates in hiring_rates_by_tier.items():
            avg_hiring_rates[tier] = sum(rates) / len(rates) if rates else 0
        
        # Calculate elite vs new advantage
        elite_rate = avg_hiring_rates.get('Elite', 0)
        new_rate = avg_hiring_rates.get('New', 0.001)  # Avoid division by zero
        elite_vs_new_ratio = elite_rate / new_rate if new_rate > 0 else 1.0
        
        # Count reputation mobility (simplified - based on tier distribution)
        freelancers_promoted = sum(tier_job_counts[tier] for tier in ['Established', 'Expert', 'Elite'])
        clients_promoted = sum(1 for client in client_data if client.get('successful_hires', 0) >= 3)
        
        return {
            'tier_hiring_rates': avg_hiring_rates,
            'tier_hiring_advantage': {
                'elite_vs_new_ratio': elite_vs_new_ratio,
                'expert_vs_new_ratio': avg_hiring_rates.get('Expert', 0) / new_rate if new_rate > 0 else 1.0
            },
            'reputation_mobility': {
                'freelancers_promoted': freelancers_promoted,
                'clients_promoted': clients_promoted
            },
            'tier_job_distribution': tier_job_counts
        }
    

    
    def _extract_reputation_history(self, round_data: List[Dict]) -> List[Dict]:
        """Extract reputation snapshots from round data"""
        reputation_history = []
        
        for round_info in round_data:
            reputation_snapshot = round_info.get('reputation_snapshot', {})
            if reputation_snapshot:
                reputation_history.append({
                    'round': round_info.get('round', 0),
                    'reputation_data': reputation_snapshot
                })
        
        return reputation_history
    
    def _analyze_reputation_progression(self, reputation_history: List[Dict], agent_type: str) -> Dict:
        """Analyze how agent reputations evolved over time"""
        if not reputation_history:
            return {}
        
        # Track individual agent progression
        agent_progressions = {}
        
        for snapshot in reputation_history:
            round_num = snapshot['round']
            agents_data = snapshot['reputation_data'].get(agent_type, {})
            
            for agent_id, reputation in agents_data.items():
                if reputation is None:
                    continue
                    
                if agent_id not in agent_progressions:
                    agent_progressions[agent_id] = []
                
                current_tier = reputation.get('tier', 'New')
                overall_score = reputation.get('overall_score', 0.0)
                
                agent_progressions[agent_id].append({
                    'round': round_num,
                    'tier': current_tier,
                    'overall_score': overall_score,
                    'completion_rate': reputation.get('job_completion_rate', 0.0) if agent_type == 'freelancers' else reputation.get('hire_success_rate', 0.0),
                    'total_experience': reputation.get('total_jobs_hired', 0) if agent_type == 'freelancers' else reputation.get('total_jobs_posted', 0)
                })
        
        # Analyze progression patterns
        learning_curves = []
        tier_mobility = 0
        
        for agent_id, progression in agent_progressions.items():
            if len(progression) > 1:
                # Check for tier advancement
                initial_tier = progression[0]['tier']
                final_tier = progression[-1]['tier']
                
                tier_hierarchy = {'New': 0, 'Established': 1, 'Expert': 2, 'Elite': 3}
                if tier_hierarchy.get(final_tier, 0) > tier_hierarchy.get(initial_tier, 0):
                    tier_mobility += 1
                
                # Track learning curve (score improvement)
                initial_score = progression[0]['overall_score']
                final_score = progression[-1]['overall_score']
                score_improvement = final_score - initial_score
                
                learning_curves.append(score_improvement)
        
        # Calculate averages
        avg_score_improvement = sum(learning_curves) / len(learning_curves) if learning_curves else 0
        mobility_rate = tier_mobility / len(agent_progressions) if agent_progressions else 0
        
        return {
            'agents_tracked': len(agent_progressions),
            'tier_mobility_rate': mobility_rate,
            'agents_promoted': tier_mobility,
            'average_score_improvement': avg_score_improvement,
            'learning_trajectory': 'improving' if avg_score_improvement > 0.1 else 'stable' if avg_score_improvement > -0.1 else 'declining',
            'progression_data': {agent_id: prog for agent_id, prog in list(agent_progressions.items())[:3]}  # Sample for display
        }
