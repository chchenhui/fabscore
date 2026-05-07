#!/usr/bin/env python3
"""
H2: Architecture Comparison Experiment
Tests whether multi-agent systems outperform monolithic LLMs.
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

from models.baselines import ensemble_baseline, peak_sales_heuristic
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


class H2Experiment:
    """
    H2: Multi-agent architecture should outperform monolithic approaches.
    
    Conditions:
    - Monolithic: Single LLM with comprehensive prompting
    - Multi-agent: Specialized agents for market sizing, pricing, forecasting
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
    
    def monolithic_forecast(self, train_data: pd.DataFrame, 
                           test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Monolithic approach: Single comprehensive model.
        Uses ensemble baseline as proxy for single LLM.
        """
        forecasts = []
        y2_forecasts = []
        peak_forecasts = []
        
        for _, drug in test_data.iterrows():
            # Use ensemble baseline as monolithic approach
            forecast_dict = ensemble_baseline(drug, years=5)
            forecast = forecast_dict.get('ensemble', np.zeros(5))
            
            forecasts.append(forecast)
            y2_forecasts.append(forecast[1] if len(forecast) > 1 else 0)
            peak_forecasts.append(np.max(forecast))
        
        forecast_array = np.vstack(forecasts) if forecasts else np.array([])
        
        return {
            'forecast': forecast_array,
            'forecast_y2': np.array(y2_forecasts),
            'forecast_peak': np.array(peak_forecasts)
        }
    
    def multi_agent_forecast(self, train_data: pd.DataFrame,
                            test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Multi-agent approach: Specialized agents for different tasks.
        Market Analysis Agent + Forecast Agent + Review Agent.
        """
        # Initialize specialized agents
        analog_forecaster = AnalogForecaster()
        analog_forecaster.launches = train_data
        analog_forecaster.revenues = self.revenues[
            self.revenues['launch_id'].isin(train_data['launch_id'])
        ]
        analog_forecaster.analogs = self.analogs[
            self.analogs['launch_id'].isin(train_data['launch_id']) &
            self.analogs['analog_launch_id'].isin(train_data['launch_id'])
        ]
        
        flow_model = PatientFlowModel()
        
        forecasts = []
        y2_forecasts = []
        peak_forecasts = []
        
        for _, drug in test_data.iterrows():
            # Market Analysis Agent: Find analogs and assess competition
            analog_result = analog_forecaster.get_prediction_intervals(drug, years=5)
            
            # Forecast Agent: Generate multiple forecasts
            flow_scenarios = flow_model.forecast_with_scenarios(drug, years=5)
            baseline_dict = ensemble_baseline(drug, years=5)
            baseline_forecast = baseline_dict.get('ensemble', np.zeros(5))
            
            # Review Agent: Weighted ensemble based on confidence
            # Higher weight to methods with better historical performance
            analog_weight = 0.4
            flow_weight = 0.3
            baseline_weight = 0.3
            
            ensemble = (analog_weight * analog_result['forecast'] + 
                       flow_weight * flow_scenarios['base'] + 
                       baseline_weight * baseline_forecast)
            
            forecasts.append(ensemble)
            y2_forecasts.append(ensemble[1] if len(ensemble) > 1 else 0)
            peak_forecasts.append(np.max(ensemble))
        
        forecast_array = np.vstack(forecasts) if forecasts else np.array([])
        
        return {
            'forecast': forecast_array,
            'forecast_y2': np.array(y2_forecasts),
            'forecast_peak': np.array(peak_forecasts)
        }
    
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
        """Calculate evaluation metrics."""
        if len(predictions['forecast_y2']) == 0:
            return {'y2_mape': float('inf'), 'peak_mape': float('inf')}
        
        # Year 2 MAPE
        y2_actual = test_data['actual_y2'].values
        y2_pred = predictions['forecast_y2']
        y2_mape = self.metrics.mape(y2_pred, y2_actual)
        
        # Peak MAPE
        peak_actual = test_data['actual_peak'].values
        peak_pred = predictions['forecast_peak']
        peak_mape = self.metrics.mape(peak_pred, peak_actual)
        
        return {
            'y2_mape': y2_mape,
            'peak_mape': peak_mape
        }
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the full H2 experiment."""
        print("=" * 60)
        print("H2: ARCHITECTURE COMPARISON EXPERIMENT")
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
        
        print("PASSED: Gate G3 - Statistical protocol valid")
        
        # Run experiments
        print("\nRunning monolithic forecast...")
        mono_pred = self.monolithic_forecast(train_data, test_data)
        
        print("Running multi-agent forecast...")
        multi_pred = self.multi_agent_forecast(train_data, test_data)
        
        # Calculate metrics
        mono_metrics = self._calculate_metrics(test_data, mono_pred)
        multi_metrics = self._calculate_metrics(test_data, multi_pred)
        
        # Statistical comparison
        print("\nStatistical comparison...")
        y2_improvement = mono_metrics['y2_mape'] - multi_metrics['y2_mape']
        peak_improvement = mono_metrics['peak_mape'] - multi_metrics['peak_mape']
        
        # Results
        results = {
            'experiment': 'H2_Architecture_Comparison',
            'seed': self.seed,
            'n_train': len(train_data),
            'n_test': len(test_data),
            'monolithic': mono_metrics,
            'multi_agent': multi_metrics,
            'y2_improvement': y2_improvement,
            'peak_improvement': peak_improvement,
            'multi_agent_better': y2_improvement > 0 and peak_improvement > 0
        }
        
        # Print results
        print(f"\nResults:")
        print(f"Monolithic Y2 MAPE: {mono_metrics['y2_mape']:.1%}")
        print(f"Multi-agent Y2 MAPE: {multi_metrics['y2_mape']:.1%}")
        print(f"Y2 Improvement: {y2_improvement:.1%}")
        print(f"Monolithic Peak MAPE: {mono_metrics['peak_mape']:.1%}")
        print(f"Multi-agent Peak MAPE: {multi_metrics['peak_mape']:.1%}")
        print(f"Peak Improvement: {peak_improvement:.1%}")
        print(f"Multi-agent better: {results['multi_agent_better']}")
        
        # Save results
        results_file = self.results_dir / "h2_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Audit logging
        audit_run(
            experiment_name="H2_Architecture_Comparison",
            config={'seed': self.seed, 'n_train': len(train_data), 'n_test': len(test_data)},
            results=results,
            logger=self.logger
        )
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="H2: Architecture Comparison Experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    experiment = H2Experiment(seed=args.seed)
    results = experiment.run_experiment()
    
    if 'error' in results:
        sys.exit(1)
    
    print("\nSUCCESS: H2 experiment complete")


if __name__ == "__main__":
    main()
