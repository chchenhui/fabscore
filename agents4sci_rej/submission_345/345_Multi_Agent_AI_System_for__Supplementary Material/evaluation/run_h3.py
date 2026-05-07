#!/usr/bin/env python3
"""
H3: Domain Constraints Experiment
Tests whether Bass diffusion constraints improve forecast accuracy.
Following Linus principle: Measure what matters, ignore what doesn't.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.baselines import ensemble_baseline, peak_sales_heuristic, simple_bass_forecast
from models.analogs import AnalogForecaster
from models.patient_flow import PatientFlowModel
from stats.protocol import (
    StatisticalProtocol, 
    EvaluationMetrics,
    CrossValidation,
    HypothesisTesting,
    AcceptanceGates
)
from utils.audit import get_audit_logger, audit_run


class H3Experiment:
    """
    H3: Domain constraints should improve forecast accuracy and prediction interval coverage.
    
    Conditions:
    - Unconstrained: LLM forecasts without domain-specific constraints
    - Constrained: Bass model with pharmaceutical domain constraints
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
        self.protocol = StatisticalProtocol(seed=seed)
        self.metrics = EvaluationMetrics()
        self.cv = CrossValidation(self.protocol)
        self.hypothesis = HypothesisTesting(self.protocol)
        self.gates = AcceptanceGates()
        self.logger = get_audit_logger()
        
        # Load data
        self.data_dir = Path(__file__).parent.parent / "data_proc"
        self.results_dir = Path(__file__).parent.parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self._load_data()
    
    def _load_data(self):
        """Load pharmaceutical launch data."""
        try:
            self.launches = pd.read_parquet(self.data_dir / "launches.parquet")
            self.revenues = pd.read_parquet(self.data_dir / "launch_revenues.parquet")
            self.analogs = pd.read_parquet(self.data_dir / "analogs.parquet")
            
            # Parse approval_date to datetime for temporal splits
            if 'approval_date' in self.launches.columns:
                self.launches['approval_date'] = pd.to_datetime(self.launches['approval_date'])
            
            print(f"Loaded {len(self.launches)} launches")
        except FileNotFoundError:
            print("ERROR: Dataset not found. Run 'python src/cli.py build-data' first")
            sys.exit(1)
    
    def unconstrained_forecast(self, train_data: pd.DataFrame, 
                              test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Unconstrained approach: No domain-specific constraints.
        Uses analog forecasting without Bass constraints.
        """
        analog_forecaster = AnalogForecaster()
        analog_forecaster.launches = train_data
        analog_forecaster.revenues = self.revenues[
            self.revenues['launch_id'].isin(train_data['launch_id'])
        ]
        analog_forecaster.analogs = self.analogs[
            self.analogs['launch_id'].isin(train_data['launch_id']) &
            self.analogs['analog_launch_id'].isin(train_data['launch_id'])
        ]
        
        forecasts = []
        y2_forecasts = []
        peak_forecasts = []
        lowers = []
        uppers = []
        
        for _, drug in test_data.iterrows():
            # Get analog forecast without constraints
            result = analog_forecaster.get_prediction_intervals(drug, years=5)
            
            forecasts.append(result['forecast'])
            y2_forecasts.append(result['forecast'][1] if len(result['forecast']) > 1 else 0)
            peak_forecasts.append(np.max(result['forecast']))
            lowers.append(result['lower'])
            uppers.append(result['upper'])
        
        forecast_array = np.vstack(forecasts) if forecasts else np.array([])
        lower_array = np.vstack(lowers) if lowers else np.array([])
        upper_array = np.vstack(uppers) if uppers else np.array([])
        
        return {
            'forecast': forecast_array,
            'forecast_y2': np.array(y2_forecasts),
            'forecast_peak': np.array(peak_forecasts),
            'lower': lower_array,
            'upper': upper_array
        }
    
    def constrained_forecast(self, train_data: pd.DataFrame,
                           test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Constrained approach: Bass diffusion with pharmaceutical constraints.
        Includes market access tiers, penetration ceilings, and adoption curves.
        """
        forecasts = []
        y2_forecasts = []
        peak_forecasts = []
        lowers = []
        uppers = []
        
        for _, drug in test_data.iterrows():
            # Get Bass forecast with constraints
            forecast = simple_bass_forecast(drug, years=5)
            
            # Apply pharmaceutical domain constraints
            constrained_forecast = self._apply_domain_constraints(forecast, drug)
            
            # Generate prediction intervals based on constraint uncertainty
            lower, upper = self._generate_constraint_intervals(constrained_forecast, drug)
            
            forecasts.append(constrained_forecast)
            y2_forecasts.append(constrained_forecast[1] if len(constrained_forecast) > 1 else 0)
            peak_forecasts.append(np.max(constrained_forecast))
            lowers.append(lower)
            uppers.append(upper)
        
        forecast_array = np.vstack(forecasts) if forecasts else np.array([])
        lower_array = np.vstack(lowers) if lowers else np.array([])
        upper_array = np.vstack(uppers) if uppers else np.array([])
        
        return {
            'forecast': forecast_array,
            'forecast_y2': np.array(y2_forecasts),
            'forecast_peak': np.array(peak_forecasts),
            'lower': lower_array,
            'upper': upper_array
        }
    
    def _apply_domain_constraints(self, forecast: np.ndarray, drug: Dict) -> np.ndarray:
        """Apply pharmaceutical domain constraints to forecast."""
        constrained = forecast.copy()
        
        # Market access constraint
        access_tier = drug.get('access_tier_at_launch', 'PA')
        if access_tier == 'NICHE':
            # Niche access: cap at 20% of unconstrained
            constrained *= 0.2
        elif access_tier == 'PA':
            # Prior authorization: cap at 60% of unconstrained
            constrained *= 0.6
        
        # Competition constraint
        competitors = drug.get('competitor_count_at_launch', 3)
        if competitors > 5:
            # High competition: reduce by 30%
            constrained *= 0.7
        elif competitors > 2:
            # Moderate competition: reduce by 15%
            constrained *= 0.85
        
        # Efficacy constraint
        efficacy = drug.get('clinical_efficacy_proxy', 0.7)
        if efficacy < 0.5:
            # Low efficacy: reduce by 40%
            constrained *= 0.6
        elif efficacy < 0.7:
            # Moderate efficacy: reduce by 20%
            constrained *= 0.8
        
        # Safety constraint
        if drug.get('safety_black_box', False):
            # Black box warning: reduce by 25%
            constrained *= 0.75
        
        return constrained
    
    def _generate_constraint_intervals(self, forecast: np.ndarray, drug: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals based on constraint uncertainty."""
        # Base uncertainty from constraints
        base_uncertainty = 0.2  # 20% base uncertainty
        
        # Additional uncertainty from access constraints
        access_tier = drug.get('access_tier_at_launch', 'PA')
        if access_tier == 'NICHE':
            access_uncertainty = 0.3  # High uncertainty for niche
        elif access_tier == 'PA':
            access_uncertainty = 0.2  # Moderate uncertainty
        else:
            access_uncertainty = 0.1  # Low uncertainty for open access
        
        # Competition uncertainty
        competitors = drug.get('competitor_count_at_launch', 3)
        competition_uncertainty = min(0.2, competitors * 0.03)  # Up to 20% for high competition
        
        # Total uncertainty
        total_uncertainty = base_uncertainty + access_uncertainty + competition_uncertainty
        
        # Generate intervals
        lower = forecast * (1 - total_uncertainty)
        upper = forecast * (1 + total_uncertainty)
        
        return lower, upper
    
    def prepare_evaluation_data(self) -> pd.DataFrame:
        """Prepare data with actual values for evaluation."""
        # Merge launches with revenues to get actuals
        y2_revenues = self.revenues[self.revenues['year_since_launch'] == 1].rename(
            columns={'revenue_usd': 'actual_y2'}
        )[['launch_id', 'actual_y2']]
        
        peak_revenues = self.revenues.groupby('launch_id')['revenue_usd'].max().reset_index()
        peak_revenues.columns = ['launch_id', 'actual_peak']
        
        # Merge with launches
        eval_data = self.launches.merge(y2_revenues, on='launch_id', how='left')
        eval_data = eval_data.merge(peak_revenues, on='launch_id', how='left')
        
        # Fill missing with estimates (for incomplete data)
        eval_data['actual_y2'] = eval_data['actual_y2'].fillna(
            eval_data.apply(lambda x: peak_sales_heuristic(x) * 0.25, axis=1)
        )
        eval_data['actual_peak'] = eval_data['actual_peak'].fillna(
            eval_data.apply(peak_sales_heuristic, axis=1)
        )
        
        return eval_data
    
    def _calculate_metrics(self, test_data: pd.DataFrame, 
                          predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate evaluation metrics including PI coverage."""
        if len(predictions['forecast_y2']) == 0:
            return {
                'y2_mape': float('inf'), 
                'peak_mape': float('inf'),
                'pi_coverage': 0.0
            }
        
        # Year 2 MAPE
        y2_actual = test_data['actual_y2'].values
        y2_pred = predictions['forecast_y2']
        y2_mape = self.metrics.mape(y2_pred, y2_actual)
        
        # Peak MAPE
        peak_actual = test_data['actual_peak'].values
        peak_pred = predictions['forecast_peak']
        peak_mape = self.metrics.mape(peak_pred, peak_actual)
        
        # Prediction interval coverage
        pi_coverage = 0.0
        if 'lower' in predictions and 'upper' in predictions:
            # Calculate coverage for each year
            coverage_by_year = []
            for year in range(min(5, predictions['forecast'].shape[1])):
                year_actual = self.revenues[
                    (self.revenues['year_since_launch'] == year) &
                    (self.revenues['launch_id'].isin(test_data['launch_id']))
                ]['revenue_usd'].values
                
                if len(year_actual) > 0:
                    year_lower = predictions['lower'][:len(year_actual), year]
                    year_upper = predictions['upper'][:len(year_actual), year]
                    year_coverage = np.mean((year_actual >= year_lower) & (year_actual <= year_upper))
                    coverage_by_year.append(year_coverage)
            
            pi_coverage = np.mean(coverage_by_year) if coverage_by_year else 0.0
        
        return {
            'y2_mape': y2_mape,
            'peak_mape': peak_mape,
            'pi_coverage': pi_coverage
        }
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the full H3 experiment."""
        print("=" * 60)
        print("H3: DOMAIN CONSTRAINTS EXPERIMENT")
        print("=" * 60)
        
        # Prepare data
        eval_data = self.prepare_evaluation_data()
        
        # Temporal split
        train_data, test_data = self.cv.split_temporal(eval_data)
        print(f"\nData split: {len(train_data)} train, {len(test_data)} test")
        
        # Check Gate G3
        g3_passed, g3_msg = self.gates.check_gate_g3(
            self.protocol, len(train_data), len(test_data)
        )
        if not g3_passed:
            print(f"ERROR: {g3_msg}")
            return {'error': g3_msg}
        
        print("✓ Gate G3 PASSED: Statistical protocol valid")
        
        # Run experiments
        print("\nRunning unconstrained forecast...")
        unconstrained_pred = self.unconstrained_forecast(train_data, test_data)
        
        print("Running constrained forecast...")
        constrained_pred = self.constrained_forecast(train_data, test_data)
        
        # Calculate metrics
        unconstrained_metrics = self._calculate_metrics(test_data, unconstrained_pred)
        constrained_metrics = self._calculate_metrics(test_data, constrained_pred)
        
        # Statistical comparison
        print("\nStatistical comparison...")
        y2_improvement = unconstrained_metrics['y2_mape'] - constrained_metrics['y2_mape']
        peak_improvement = unconstrained_metrics['peak_mape'] - constrained_metrics['peak_mape']
        pi_improvement = constrained_metrics['pi_coverage'] - unconstrained_metrics['pi_coverage']
        
        # Results
        results = {
            'experiment': 'H3_Domain_Constraints',
            'seed': self.seed,
            'n_train': len(train_data),
            'n_test': len(test_data),
            'unconstrained': unconstrained_metrics,
            'constrained': constrained_metrics,
            'y2_improvement': y2_improvement,
            'peak_improvement': peak_improvement,
            'pi_coverage_improvement': pi_improvement,
            'constraints_help': y2_improvement > 0 and peak_improvement > 0 and pi_improvement > 0
        }
        
        # Print results
        print(f"\nResults:")
        print(f"Unconstrained Y2 MAPE: {unconstrained_metrics['y2_mape']:.1%}")
        print(f"Constrained Y2 MAPE: {constrained_metrics['y2_mape']:.1%}")
        print(f"Y2 Improvement: {y2_improvement:.1%}")
        print(f"Unconstrained Peak MAPE: {unconstrained_metrics['peak_mape']:.1%}")
        print(f"Constrained Peak MAPE: {constrained_metrics['peak_mape']:.1%}")
        print(f"Peak Improvement: {peak_improvement:.1%}")
        print(f"Unconstrained PI Coverage: {unconstrained_metrics['pi_coverage']:.1%}")
        print(f"Constrained PI Coverage: {constrained_metrics['pi_coverage']:.1%}")
        print(f"PI Coverage Improvement: {pi_improvement:.1%}")
        print(f"Constraints help: {results['constraints_help']}")
        
        # Save results
        results_file = self.results_dir / "h3_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Audit logging
        audit_run(
            experiment_name="H3_Domain_Constraints",
            config={'seed': self.seed, 'n_train': len(train_data), 'n_test': len(test_data)},
            results=results,
            logger=self.logger
        )
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="H3: Domain Constraints Experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    experiment = H3Experiment(seed=args.seed)
    results = experiment.run_experiment()
    
    if 'error' in results:
        sys.exit(1)
    
    print("\n✓ H3 experiment complete")


if __name__ == "__main__":
    main()
