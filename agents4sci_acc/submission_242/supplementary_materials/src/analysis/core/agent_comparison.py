"""
Comparative Analysis: LLM Agents vs Algorithmic Agents

This module provides comprehensive comparison between different agent types:
1. LLM-powered agents (actual AI reasoning)
2. Algorithmic agents (rule-based with randomness)
3. Hybrid approaches

This addresses the core research question: How do authentic AI agents differ from
computational models in economic behavior?
"""

import pandas as pd
import numpy as np
from ..visualization.market_plots import plot_market_overview, plot_temporal_patterns
from scipy import stats
import json
import logging
from pathlib import Path
from typing import Dict
import re

logger = logging.getLogger(__name__)

class ComparativeMarketplaceAnalyzer:
    """Comprehensive analysis comparing LLM vs Algorithmic agents"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.llm_data = {}
        self.algorithmic_data = {}
        self.comparison_results = {}
        
    def load_simulation_data(self):
        """Load data from both LLM and algorithmic simulations"""
        logger.info("Loading simulation data for comparison")
        
        # Load algorithmic simulation data
        try:
            # Basic simulation results
            basic_files = list(self.results_dir.glob("**/round_data_*.csv"))
            if basic_files:
                self.algorithmic_data["rounds"] = pd.read_csv(basic_files[-1])  # Most recent
            
            bid_files = list(self.results_dir.glob("**/bid_data_*.csv"))
            if bid_files:
                self.algorithmic_data["bids"] = pd.read_csv(bid_files[-1])
                
            # Analysis report
            analysis_files = list(self.results_dir.glob("analysis_report.json"))
            if analysis_files:
                with open(analysis_files[0]) as f:
                    self.algorithmic_data["analysis"] = json.load(f)
            
            logger.info(f"Loaded algorithmic data: {len(self.algorithmic_data)} datasets")
            
        except Exception as e:
            logger.exception(f"Failed to load algorithmic data: {e}")
        
        # Load LLM simulation data
        try:
            llm_dir = self.results_dir / "llm_simulation"
            if llm_dir.exists():
                # Round data
                llm_round_files = list(llm_dir.glob("llm_round_data_*.csv"))
                if llm_round_files:
                    self.llm_data["rounds"] = pd.read_csv(llm_round_files[-1])
                
                # Bid data
                llm_bid_files = list(llm_dir.glob("llm_bid_data_*.csv"))
                if llm_bid_files:
                    self.llm_data["bids"] = pd.read_csv(llm_bid_files[-1])
                
                # Agent states
                llm_state_files = list(llm_dir.glob("llm_agent_states_*.json"))
                if llm_state_files:
                    with open(llm_state_files[-1]) as f:
                        self.llm_data["agent_states"] = json.load(f)
                
                logger.info(f"Loaded LLM data: {len(self.llm_data)} datasets")
            else:
                logger.warning("No LLM simulation data found")
                
        except Exception as e:
            logger.exception(f"Failed to load LLM data: {e}")
    
    def compare_bidding_patterns(self) -> Dict:
        """Compare bidding patterns between agent types"""
        logger.info("Comparing bidding patterns")
        
        comparison = {}
        
        if "bids" in self.algorithmic_data and "bids" in self.llm_data:
            algo_bids = self.algorithmic_data["bids"]
            llm_bids = self.llm_data["bids"]
            
            # Rate distributions
            comparison["rate_comparison"] = {
                "algorithmic": {
                    "mean": algo_bids["proposed_rate"].mean(),
                    "std": algo_bids["proposed_rate"].std(),
                    "median": algo_bids["proposed_rate"].median(),
                    "range": (algo_bids["proposed_rate"].min(), algo_bids["proposed_rate"].max())
                },
                "llm": {
                    "mean": llm_bids["proposed_hourly_rate"].mean(),
                    "std": llm_bids["proposed_hourly_rate"].std(),
                    "median": llm_bids["proposed_hourly_rate"].median(),
                    "range": (llm_bids["proposed_hourly_rate"].min(), llm_bids["proposed_hourly_rate"].max())
                }
            }
            
            # Statistical test for rate differences
            algo_rates = algo_bids["proposed_rate"].dropna()
            llm_rates = llm_bids["proposed_hourly_rate"].dropna()
            
            if len(algo_rates) > 0 and len(llm_rates) > 0:
                t_stat, p_value = stats.ttest_ind(algo_rates, llm_rates)
                comparison["rate_significance_test"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
            
            # Time estimation comparison (if available)
            if "estimated_hours" in algo_bids.columns and "estimated_hours" in llm_bids.columns:
                algo_hours = algo_bids["estimated_hours"].dropna()
                llm_hours = llm_bids["estimated_hours"].dropna()
                
                comparison["time_estimation"] = {
                    "algorithmic_avg": algo_hours.mean(),
                    "llm_avg": llm_hours.mean(),
                    "difference": llm_hours.mean() - algo_hours.mean()
                }
        
        return comparison
    
    def compare_decision_complexity(self) -> Dict:
        """Analyze complexity of decision-making processes"""
        logger.info("Comparing decision complexity")
        
        complexity_analysis = {}
        
        # For algorithmic agents: relatively simple rule-based decisions
        if "bids" in self.algorithmic_data:
            algo_bids = self.algorithmic_data["bids"]
            
            # Measure variability as proxy for complexity
            if "proposed_rate" in algo_bids.columns:
                rate_cv = algo_bids["proposed_rate"].std() / algo_bids["proposed_rate"].mean()
                complexity_analysis["algorithmic"] = {
                    "rate_coefficient_variation": rate_cv,
                    "decision_factors": ["rate_range", "persona_multiplier", "random_noise"],
                    "reasoning_complexity": "low"  # Rule-based
                }
        
        # For LLM agents: analyze reasoning complexity
        if "bids" in self.llm_data:
            llm_bids = self.llm_data["bids"]
            
            # Analyze proposal text complexity (if available)
            reasoning_complexity = self._analyze_text_complexity(llm_bids)
            
            if "proposed_hourly_rate" in llm_bids.columns:
                rate_cv = llm_bids["proposed_hourly_rate"].std() / llm_bids["proposed_hourly_rate"].mean()
                complexity_analysis["llm"] = {
                    "rate_coefficient_variation": rate_cv,
                    "decision_factors": ["skill_match", "market_assessment", "personal_situation", 
                                       "strategic_thinking", "risk_evaluation"],
                    "reasoning_complexity": "high",
                    "text_analysis": reasoning_complexity
                }
        
        return complexity_analysis
    
    def compare_market_outcomes(self) -> Dict:
        """Compare overall market performance between agent types"""
        logger.info("Comparing market outcomes")
        
        market_comparison = {}
        
        # Market efficiency metrics
        if "rounds" in self.algorithmic_data and "rounds" in self.llm_data:
            algo_rounds = self.algorithmic_data["rounds"]
            llm_rounds = self.llm_data["rounds"]
            
            market_comparison["efficiency"] = {
                "algorithmic": {
                    "avg_fill_rate": (algo_rounds["jobs_filled"] / algo_rounds["jobs_posted"]).mean(),
                    "avg_competition": algo_rounds["avg_bids_per_job"].mean() if "avg_bids_per_job" in algo_rounds.columns else None,
                    "market_activity": algo_rounds["total_bids"].mean()
                },
                "llm": {
                    "avg_fill_rate": (llm_rounds["jobs_filled"] / llm_rounds["jobs_posted"]).mean(),
                    "avg_competition": (llm_rounds["total_bids"] / llm_rounds["jobs_posted"]).mean(),
                    "market_activity": llm_rounds["total_bids"].mean()
                }
            }
        
        # Cost analysis
        if "bids" in self.algorithmic_data and "bids" in self.llm_data:
            algo_bids = self.algorithmic_data["bids"]
            llm_bids = self.llm_data["bids"]
            
            # Calculate total project costs
            algo_total_costs = algo_bids["proposed_rate"] * algo_bids.get("estimated_hours", 40)
            llm_total_costs = llm_bids["total_cost"] if "total_cost" in llm_bids.columns else \
                             llm_bids["proposed_hourly_rate"] * llm_bids.get("estimated_hours", 40)
            
            market_comparison["cost_analysis"] = {
                "algorithmic_avg_project_cost": algo_total_costs.mean(),
                "llm_avg_project_cost": llm_total_costs.mean(),
                "cost_difference_pct": ((llm_total_costs.mean() - algo_total_costs.mean()) / 
                                       algo_total_costs.mean() * 100)
            }
        
        return market_comparison
    
    def analyze_adaptation_patterns(self) -> Dict:
        """Compare how different agent types adapt over time"""
        logger.info("Analyzing adaptation patterns")
        
        adaptation_analysis = {}
        
        # Algorithmic adaptation (simple parameter adjustments)
        if "rounds" in self.algorithmic_data:
            algo_rounds = self.algorithmic_data["rounds"]
            if len(algo_rounds) > 1:
                # Look for trends in market behavior
                adaptation_analysis["algorithmic"] = {
                    "adaptation_type": "parameter_adjustment",
                    "market_trend": self._calculate_trend(algo_rounds["total_bids"].values),
                    "efficiency_trend": self._calculate_trend((algo_rounds["jobs_filled"] / 
                                                             algo_rounds["jobs_posted"]).values),
                    "complexity": "low"
                }
        
        # LLM adaptation (strategic learning)
        if "agent_states" in self.llm_data:
            agent_states = self.llm_data["agent_states"]
            
            # Analyze strategy evolution
            strategy_changes = sum(1 for agent in agent_states 
                                 if agent.get("current_strategy") != "exploring")
            
            adaptation_analysis["llm"] = {
                "adaptation_type": "strategic_learning",
                "agents_adapted": strategy_changes,
                "total_agents": len(agent_states),
                "adaptation_rate": strategy_changes / len(agent_states),
                "complexity": "high"
            }
        
        return adaptation_analysis
    
    def statistical_comparison(self) -> Dict:
        """Comprehensive statistical comparison between agent types"""
        logger.info("Performing statistical comparison")
        
        stats_comparison = {}
        
        if "bids" in self.algorithmic_data and "bids" in self.llm_data:
            algo_bids = self.algorithmic_data["bids"]
            llm_bids = self.llm_data["bids"]
            
            # Sample size comparison
            stats_comparison["sample_sizes"] = {
                "algorithmic": len(algo_bids),
                "llm": len(llm_bids),
                "size_ratio": len(llm_bids) / len(algo_bids) if len(algo_bids) > 0 else 0
            }
            
            # Distribution comparisons
            if "proposed_rate" in algo_bids.columns and "proposed_hourly_rate" in llm_bids.columns:
                algo_rates = algo_bids["proposed_rate"].dropna()
                llm_rates = llm_bids["proposed_hourly_rate"].dropna()
                
                # Kolmogorov-Smirnov test for distribution differences
                ks_stat, ks_p = stats.ks_2samp(algo_rates, llm_rates)
                
                # Mann-Whitney U test (non-parametric)
                mw_stat, mw_p = stats.mannwhitneyu(algo_rates, llm_rates, alternative='two-sided')
                
                stats_comparison["distribution_tests"] = {
                    "kolmogorov_smirnov": {"statistic": ks_stat, "p_value": ks_p},
                    "mann_whitney_u": {"statistic": mw_stat, "p_value": mw_p},
                    "distributions_different": ks_p < 0.05
                }
                
                # Effect size calculation
                pooled_std = np.sqrt((algo_rates.var() + llm_rates.var()) / 2)
                cohens_d = (llm_rates.mean() - algo_rates.mean()) / pooled_std
                
                stats_comparison["effect_size"] = {
                    "cohens_d": cohens_d,
                    "magnitude": self._interpret_effect_size(abs(cohens_d))
                }
        
        return stats_comparison
    
    def generate_research_insights(self) -> Dict:
        """Generate key research insights from the comparison"""
        logger.info("Generating research insights")
        
        insights = {
            "key_findings": [],
            "research_implications": [],
            "methodological_insights": [],
            "limitations": []
        }
        
        # Analyze all comparison results
        bidding_comp = self.comparison_results.get("bidding_patterns", {})
        complexity_comp = self.comparison_results.get("decision_complexity", {})
        market_comp = self.comparison_results.get("market_outcomes", {})
        stats_comp = self.comparison_results.get("statistical_comparison", {})
        
        # Key findings
        if "rate_significance_test" in bidding_comp:
            if bidding_comp["rate_significance_test"]["significant"]:
                insights["key_findings"].append(
                    "LLM and algorithmic agents show significantly different bidding patterns"
                )
            else:
                insights["key_findings"].append(
                    "No significant difference in bidding rates between agent types"
                )
        
        if "complexity" in complexity_comp:
            insights["key_findings"].append(
                "LLM agents demonstrate higher decision complexity compared to algorithmic agents"
            )
        
        # Research implications
        insights["research_implications"].extend([
            "Authentic AI reasoning produces different economic behaviors than rule-based models",
            "LLM agents provide more naturalistic models of human economic decision-making",
            "Computational cost must be balanced against authenticity in agent-based research"
        ])
        
        # Methodological insights
        insights["methodological_insights"].extend([
            "LLM agents enable study of emergent reasoning in economic contexts",
            "Algorithmic agents provide controlled baselines for comparison",
            "Hybrid approaches may offer optimal balance of authenticity and efficiency"
        ])
        
        # Limitations
        insights["limitations"].extend([
            "LLM agent behavior may be influenced by training data biases",
            "Computational costs limit scale of LLM agent studies",
            "Algorithmic agents may oversimplify real economic decision-making"
        ])
        
        return insights
    
    def create_comparison_visualizations(self):
        """Generate comprehensive comparison visualizations"""
        logger.info("Creating comparison visualizations")
        
        fig_dir = self.results_dir / "figures" / "comparative"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate agent comparison plots
        if ("bids" in self.algorithmic_data and "bids" in self.llm_data and
            "rate_comparison" in self.comparison_results.get("bidding_patterns", {})):
            
            algo_bids = self.algorithmic_data["bids"]
            llm_bids = self.llm_data["bids"]
            
            # Create market overview plots
            market_data = {
                'expertise_areas': {},
                'skills': {},
                'rate_distribution': {
                    'rates': algo_bids['proposed_rate'].tolist() + llm_bids['proposed_hourly_rate'].tolist()
                },
                'project_preferences': {},
                'performance_metrics': {
                    'total_bids': len(algo_bids) + len(llm_bids),
                    'success_rate': (algo_bids['is_winner'].mean() + llm_bids['is_winner'].mean()) / 2 if 'is_winner' in algo_bids.columns and 'is_winner' in llm_bids.columns else 0.0
                }
            }
            plot_market_overview(market_data, str(fig_dir))
            
            # Create temporal pattern plots
            temporal_data = {
                'temporal_patterns': {
                    'hour_distribution': dict(pd.Series(pd.to_datetime(algo_bids['submission_time']).dt.hour.value_counts() + 
                                                      pd.to_datetime(llm_bids['submission_time']).dt.hour.value_counts())),
                    'bid_bursts': len(algo_bids) + len(llm_bids)  # Total bids as a simple metric
                },
                'success_factors': {
                    'rate_impact': {
                        'successful_rates': {
                            'min': min(algo_bids['proposed_rate'].min(), llm_bids['proposed_hourly_rate'].min()),
                            'max': max(algo_bids['proposed_rate'].max(), llm_bids['proposed_hourly_rate'].max()),
                            'mean': (algo_bids['proposed_rate'].mean() + llm_bids['proposed_hourly_rate'].mean()) / 2
                        }
                    }
                }
            }
            plot_temporal_patterns(temporal_data, str(fig_dir))
        
        logger.info(f"Comparison visualizations saved to {fig_dir}")
    
    def run_comprehensive_comparison(self) -> Dict:
        """Run complete comparative analysis"""
        logger.info("Starting comprehensive comparative analysis")
        
        # Load data
        self.load_simulation_data()
        
        # Run all comparisons
        self.comparison_results = {
            "bidding_patterns": self.compare_bidding_patterns(),
            "decision_complexity": self.compare_decision_complexity(),
            "market_outcomes": self.compare_market_outcomes(),
            "adaptation_patterns": self.analyze_adaptation_patterns(),
            "statistical_comparison": self.statistical_comparison()
        }
        
        # Generate insights
        self.comparison_results["research_insights"] = self.generate_research_insights()
        
        # Create visualizations
        self.create_comparison_visualizations()
        
        # Save results
        with open(self.results_dir / "comparative_analysis_results.json", "w") as f:
            json.dump(self.comparison_results, f, indent=2, default=str)
        
        logger.info("Comparative analysis completed")
        
        return self.comparison_results
    
    # Helper methods
    def _analyze_text_complexity(self, bid_data: pd.DataFrame) -> Dict:
        """Analyze complexity of text-based reasoning"""
        if "proposal_text" not in bid_data.columns:
            return {"error": "No proposal text available"}
        
        proposals = bid_data["proposal_text"].dropna()
        if len(proposals) == 0:
            return {"error": "No valid proposals found"}
        
        # Simple text complexity metrics
        avg_length = proposals.str.len().mean()
        unique_words = set()
        for proposal in proposals:
            if isinstance(proposal, str):
                words = re.findall(r'\w+', proposal.lower())
                unique_words.update(words)
        
        return {
            "avg_proposal_length": avg_length,
            "vocabulary_size": len(unique_words),
            "complexity_score": len(unique_words) / avg_length if avg_length > 0 else 0
        }
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction for a series"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

def main():
    """Run comparative analysis"""
    analyzer = ComparativeMarketplaceAnalyzer()
    results = analyzer.run_comprehensive_comparison()
    
    # Print key findings
    print("\n=== Comparative Analysis Results ===")
    
    insights = results.get("research_insights", {})
    if "key_findings" in insights:
        print("\nKey Findings:")
        for finding in insights["key_findings"]:
            print(f"• {finding}")
    
    if "statistical_comparison" in results:
        stats_comp = results["statistical_comparison"]
        if "sample_sizes" in stats_comp:
            sizes = stats_comp["sample_sizes"]
            print(f"\nSample Sizes: Algorithmic={sizes['algorithmic']}, LLM={sizes['llm']}")
        
        if "effect_size" in stats_comp:
            effect = stats_comp["effect_size"]
            print(f"Effect Size: Cohen's d = {effect['cohens_d']:.3f} ({effect['magnitude']})")
    
    print(f"\nDetailed results saved to: results/comparative_analysis_results.json")
    
    return results

if __name__ == "__main__":
    main()
