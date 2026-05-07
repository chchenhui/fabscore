"""
Results evaluation framework for pharmaceutical forecasting.
Tests ensemble performance against individual baselines.
Following Linus principle: Prove it works or it's garbage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import json
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.baselines import (
    peak_sales_heuristic,
    year2_naive,
    linear_trend_forecast,
    market_share_evolution,
    simple_bass_forecast,
    ensemble_baseline
)
from models.ensemble import EnsembleForecaster, EnsembleConfig
from experiment_protocol import EvaluationMetrics, StatisticalProtocol, ExperimentConfig


@dataclass
class G4GateCriteria:
    """G4 Results Gate criteria."""
    
    # Performance thresholds
    min_launches_beat_baseline: float = 0.60  # 60% of launches
    max_median_y2_ape: float = 30.0  # 30% median Year 2 APE
    min_pi_coverage: float = 0.70  # 70% prediction interval coverage
    max_pi_coverage: float = 0.90  # 90% prediction interval coverage
    
    # Statistical significance
    min_sample_size: int = 50
    confidence_level: float = 0.95
    
    # Baseline comparison
    baseline_methods: List[str] = None
    
    def __post_init__(self):
        if self.baseline_methods is None:
            self.baseline_methods = [
                'peak_sales_heuristic',
                'year2_naive',
                'linear_trend_forecast',
                'market_share_evolution',
                'simple_bass_forecast'
            ]


class ResultsEvaluator:
    """
    Comprehensive results evaluation for pharmaceutical forecasting.
    Tests ensemble performance against baselines and validates G4 gate criteria.
    """
    
    def __init__(self, data_dir: Path = None, results_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "data_proc"
        self.results_dir = results_dir or Path(__file__).parent.parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.metrics = EvaluationMetrics()
        self.protocol = StatisticalProtocol()
        self.ensemble_forecaster = EnsembleForecaster()
        
        # Results storage
        self.evaluation_results = {}
        self.g4_gate_status = {}
    
    def evaluate_ensemble_vs_baselines(self, test_data: pd.DataFrame, 
                                     actual_revenues: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate ensemble performance against individual baselines.
        
        Args:
            test_data: Test drug launches
            actual_revenues: Actual revenue data
        
        Returns:
            Comprehensive evaluation results
        """
        print("Evaluating ensemble vs baselines...")
        
        results = {
            'ensemble_results': [],
            'baseline_results': {method: [] for method in G4GateCriteria().baseline_methods},
            'comparison_metrics': {},
            'g4_gate_assessment': {}
        }
        
        # Evaluate each test drug
        for _, drug_row in test_data.iterrows():
            launch_id = drug_row['launch_id']
            
            # Get actual revenues
            drug_actuals = actual_revenues[actual_revenues['launch_id'] == launch_id]
            if drug_actuals.empty:
                continue
            
            # Convert to array format
            actual_array = self._revenues_to_array(drug_actuals)
            if actual_array is None:
                continue
            
            # Generate ensemble forecast
            try:
                ensemble_result = self.ensemble_forecaster.forecast(drug_row, years=5)
                ensemble_forecast = ensemble_result['forecast']
                ensemble_ci = ensemble_result['confidence_intervals']
                
                # Calculate ensemble metrics
                ensemble_metrics = self.metrics.calculate_all_metrics(
                    ensemble_forecast, actual_array,
                    ensemble_ci.get('lower'), ensemble_ci.get('upper')
                )
                ensemble_metrics['launch_id'] = launch_id
                results['ensemble_results'].append(ensemble_metrics)
                
            except Exception as e:
                print(f"Ensemble forecast failed for {launch_id}: {e}")
                continue
            
            # Generate baseline forecasts
            for method in G4GateCriteria().baseline_methods:
                try:
                    baseline_forecast = self._generate_baseline_forecast(drug_row, method)
                    baseline_metrics = self.metrics.calculate_all_metrics(
                        baseline_forecast, actual_array
                    )
                    baseline_metrics['launch_id'] = launch_id
                    results['baseline_results'][method].append(baseline_metrics)
                    
                except Exception as e:
                    print(f"Baseline {method} failed for {launch_id}: {e}")
                    continue
        
        # Aggregate results
        results['comparison_metrics'] = self._calculate_comparison_metrics(results)
        results['g4_gate_assessment'] = self._assess_g4_gate_criteria(results)
        
        # Save results
        self._save_evaluation_results(results)
        
        return results
    
    def _revenues_to_array(self, revenues_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Convert revenue DataFrame to array format."""
        if revenues_df.empty:
            return None
        
        # Sort by year_since_launch
        revenues_df = revenues_df.sort_values('year_since_launch')
        
        # Create array for years 0-4
        revenue_array = np.zeros(5)
        for _, row in revenues_df.iterrows():
            year = int(row['year_since_launch'])
            if 0 <= year < 5:
                revenue_array[year] = row['revenue_usd']
        
        return revenue_array
    
    def _generate_baseline_forecast(self, drug_row: pd.Series, method: str) -> np.ndarray:
        """Generate forecast using specified baseline method."""
        
        if method == 'peak_sales_heuristic':
            return linear_trend_forecast(drug_row, years=5)
        elif method == 'year2_naive':
            return self._year2_anchored_forecast(drug_row, years=5)
        elif method == 'linear_trend_forecast':
            return linear_trend_forecast(drug_row, years=5)
        elif method == 'market_share_evolution':
            share_curve = market_share_evolution(drug_row, years=5)
            market_size = drug_row['eligible_patients_at_launch']
            annual_price = drug_row['list_price_month_usd_launch'] * 12
            gtn = drug_row['net_gtn_pct_launch']
            compliance = 0.70
            return share_curve * market_size * annual_price * gtn * compliance
        elif method == 'simple_bass_forecast':
            return simple_bass_forecast(drug_row, years=5)
        else:
            raise ValueError(f"Unknown baseline method: {method}")
    
    def _year2_anchored_forecast(self, drug_row: pd.Series, years: int) -> np.ndarray:
        """Create forecast anchored to year 2 naive estimate."""
        y2_estimate = year2_naive(drug_row)
        peak = peak_sales_heuristic(drug_row)
        
        forecast = np.zeros(years)
        
        for i in range(years):
            if i == 0:
                forecast[i] = peak * 0.05
            elif i == 1:
                forecast[i] = y2_estimate
            elif i < 4:
                forecast[i] = y2_estimate + (peak - y2_estimate) * (i - 1) / 2
            else:
                forecast[i] = peak
        
        return forecast
    
    def _calculate_comparison_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comparison metrics between ensemble and baselines."""
        
        comparison = {
            'ensemble_performance': {},
            'baseline_performance': {},
            'ensemble_vs_baseline': {},
            'statistical_tests': {}
        }
        
        # Aggregate ensemble performance
        ensemble_results = results['ensemble_results']
        if ensemble_results:
            comparison['ensemble_performance'] = self._aggregate_metrics(ensemble_results)
        
        # Aggregate baseline performance
        for method, method_results in results['baseline_results'].items():
            if method_results:
                comparison['baseline_performance'][method] = self._aggregate_metrics(method_results)
        
        # Compare ensemble vs best baseline
        if ensemble_results and any(results['baseline_results'].values()):
            best_baseline = self._find_best_baseline(comparison['baseline_performance'])
            comparison['ensemble_vs_baseline'] = self._compare_ensemble_vs_baseline(
                ensemble_results, results['baseline_results'][best_baseline]
            )
        
        return comparison
    
    def _aggregate_metrics(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple results."""
        if not results:
            return {}
        
        metrics = ['mape', 'year2_ape', 'peak_ape', 'directional_accuracy', 'pi_coverage']
        aggregated = {}
        
        for metric in metrics:
            values = [r.get(metric, np.nan) for r in results if metric in r]
            if values:
                aggregated[f'{metric}_mean'] = np.nanmean(values)
                aggregated[f'{metric}_median'] = np.nanmedian(values)
                aggregated[f'{metric}_std'] = np.nanstd(values)
                aggregated[f'{metric}_count'] = len([v for v in values if not np.isnan(v)])
        
        return aggregated
    
    def _find_best_baseline(self, baseline_performance: Dict[str, Dict[str, float]]) -> str:
        """Find the best performing baseline method."""
        if not baseline_performance:
            return None
        
        best_method = None
        best_mape = float('inf')
        
        for method, metrics in baseline_performance.items():
            mape = metrics.get('mape_median', float('inf'))
            if mape < best_mape:
                best_mape = mape
                best_method = method
        
        return best_method
    
    def _compare_ensemble_vs_baseline(self, ensemble_results: List[Dict[str, float]], 
                                    baseline_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Compare ensemble vs baseline performance."""
        
        # Align results by launch_id
        ensemble_dict = {r['launch_id']: r for r in ensemble_results}
        baseline_dict = {r['launch_id']: r for r in baseline_results}
        
        common_launches = set(ensemble_dict.keys()) & set(baseline_dict.keys())
        
        if not common_launches:
            return {'error': 'No common launches found'}
        
        # Calculate wins/losses
        ensemble_wins = 0
        baseline_wins = 0
        ties = 0
        
        mape_differences = []
        
        for launch_id in common_launches:
            ensemble_mape = ensemble_dict[launch_id].get('mape', np.nan)
            baseline_mape = baseline_dict[launch_id].get('mape', np.nan)
            
            if np.isnan(ensemble_mape) or np.isnan(baseline_mape):
                continue
            
            mape_diff = baseline_mape - ensemble_mape  # Positive means ensemble is better
            mape_differences.append(mape_diff)
            
            if mape_diff > 0.01:  # Ensemble better by >1%
                ensemble_wins += 1
            elif mape_diff < -0.01:  # Baseline better by >1%
                baseline_wins += 1
            else:
                ties += 1
        
        # Statistical test
        if len(mape_differences) > 1:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(mape_differences, 0)
        else:
            t_stat, p_value = np.nan, np.nan
        
        return {
            'total_comparisons': len(common_launches),
            'ensemble_wins': ensemble_wins,
            'baseline_wins': baseline_wins,
            'ties': ties,
            'win_rate': ensemble_wins / len(common_launches) if common_launches else 0,
            'mean_mape_improvement': np.mean(mape_differences) if mape_differences else 0,
            'median_mape_improvement': np.median(mape_differences) if mape_differences else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05 if not np.isnan(p_value) else False
        }
    
    def _assess_g4_gate_criteria(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess G4 gate criteria."""
        
        criteria = G4GateCriteria()
        assessment = {
            'criteria_met': {},
            'overall_status': 'PASS',
            'details': {}
        }
        
        # Get ensemble performance
        ensemble_performance = results.get('comparison_metrics', {}).get('ensemble_performance', {})
        ensemble_vs_baseline = results.get('comparison_metrics', {}).get('ensemble_vs_baseline', {})
        
        # Criterion 1: Beat baseline on ≥60% of launches
        win_rate = ensemble_vs_baseline.get('win_rate', 0)
        criteria_1_met = win_rate >= criteria.min_launches_beat_baseline
        assessment['criteria_met']['beat_baseline_60pct'] = criteria_1_met
        assessment['details']['win_rate'] = win_rate
        assessment['details']['required_win_rate'] = criteria.min_launches_beat_baseline
        
        # Criterion 2: Median Year 2 APE ≤30%
        median_y2_ape = ensemble_performance.get('year2_ape_median', float('inf'))
        criteria_2_met = median_y2_ape <= criteria.max_median_y2_ape
        assessment['criteria_met']['median_y2_ape_30pct'] = criteria_2_met
        assessment['details']['median_y2_ape'] = median_y2_ape
        assessment['details']['max_median_y2_ape'] = criteria.max_median_y2_ape
        
        # Criterion 3: PI coverage 70-90%
        pi_coverage = ensemble_performance.get('pi_coverage_mean', 0)
        criteria_3_met = criteria.min_pi_coverage <= pi_coverage <= criteria.max_pi_coverage
        assessment['criteria_met']['pi_coverage_70_90pct'] = criteria_3_met
        assessment['details']['pi_coverage'] = pi_coverage
        assessment['details']['min_pi_coverage'] = criteria.min_pi_coverage
        assessment['details']['max_pi_coverage'] = criteria.max_pi_coverage
        
        # Overall status
        all_criteria_met = all(assessment['criteria_met'].values())
        assessment['overall_status'] = 'PASS' if all_criteria_met else 'FAIL'
        
        return assessment
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"g4_evaluation_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def run_g4_gate_evaluation(self, test_data: pd.DataFrame, 
                             actual_revenues: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete G4 gate evaluation.
        
        Args:
            test_data: Test drug launches
            actual_revenues: Actual revenue data
        
        Returns:
            G4 gate evaluation results
        """
        print("Running G4 Gate Evaluation...")
        print("=" * 50)
        
        # Check sample size
        if len(test_data) < G4GateCriteria().min_sample_size:
            print(f"WARNING: Sample size {len(test_data)} < minimum required {G4GateCriteria().min_sample_size}")
        
        # Run evaluation
        results = self.evaluate_ensemble_vs_baselines(test_data, actual_revenues)
        
        # Print summary
        self._print_evaluation_summary(results)
        
        return results
    
    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\nG4 GATE EVALUATION SUMMARY")
        print("=" * 50)
        
        # Ensemble performance
        ensemble_perf = results.get('comparison_metrics', {}).get('ensemble_performance', {})
        if ensemble_perf:
            print(f"Ensemble Performance:")
            print(f"  Median MAPE: {ensemble_perf.get('mape_median', 'N/A'):.1f}%")
            print(f"  Median Year 2 APE: {ensemble_perf.get('year2_ape_median', 'N/A'):.1f}%")
            print(f"  Median Peak APE: {ensemble_perf.get('peak_ape_median', 'N/A'):.1f}%")
            print(f"  PI Coverage: {ensemble_perf.get('pi_coverage_mean', 'N/A'):.1%}")
        
        # Ensemble vs baseline
        comparison = results.get('comparison_metrics', {}).get('ensemble_vs_baseline', {})
        if comparison:
            print(f"\nEnsemble vs Best Baseline:")
            print(f"  Win Rate: {comparison.get('win_rate', 0):.1%}")
            print(f"  Ensemble Wins: {comparison.get('ensemble_wins', 0)}")
            print(f"  Baseline Wins: {comparison.get('baseline_wins', 0)}")
            print(f"  Mean MAPE Improvement: {comparison.get('mean_mape_improvement', 0):.1f}%")
            print(f"  Statistically Significant: {comparison.get('statistically_significant', False)}")
        
        # G4 Gate status
        g4_status = results.get('g4_gate_assessment', {})
        print(f"\nG4 GATE STATUS: {g4_status.get('overall_status', 'UNKNOWN')}")
        
        criteria = g4_status.get('criteria_met', {})
        print(f"Criteria Met:")
        print(f"  Beat baseline ≥60%: {'✅' if criteria.get('beat_baseline_60pct', False) else '❌'}")
        print(f"  Median Y2 APE ≤30%: {'✅' if criteria.get('median_y2_ape_30pct', False) else '❌'}")
        print(f"  PI Coverage 70-90%: {'✅' if criteria.get('pi_coverage_70_90pct', False) else '❌'}")


def run_g4_gate_evaluation(test_data: pd.DataFrame, actual_revenues: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to run G4 gate evaluation.
    
    Args:
        test_data: Test drug launches
        actual_revenues: Actual revenue data
    
    Returns:
        G4 gate evaluation results
    """
    evaluator = ResultsEvaluator()
    return evaluator.run_g4_gate_evaluation(test_data, actual_revenues)


if __name__ == "__main__":
    # Example usage
    print("G4 Gate Evaluation System")
    print("=" * 40)
    
    # This would typically load real data
    print("Ready to evaluate ensemble vs baselines!")
    print("Use run_g4_gate_evaluation(test_data, actual_revenues) to run evaluation.")
