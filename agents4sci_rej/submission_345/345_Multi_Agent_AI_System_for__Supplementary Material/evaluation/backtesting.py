#!/usr/bin/env python3
"""
Historical backtesting framework for pharmaceutical forecasts.
Following Linus principle: Past performance matters. Test on real history.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.baselines import ensemble_baseline
from models.analogs import AnalogForecaster
from models.patient_flow import PatientFlowModel
from stats.protocol import EvaluationMetrics, StatisticalProtocol
from utils.audit import get_audit_logger, audit_run


class BacktestingFramework:
    """
    Walk-forward backtesting for pharmaceutical forecasts.
    Tests models on historical data as if forecasting in real-time.
    """
    
    def __init__(self, start_year: int = 2015, end_year: int = 2020, seed: int = 42):
        self.start_year = start_year
        self.end_year = end_year
        self.seed = seed
        np.random.seed(seed)
        
        self.metrics = EvaluationMetrics()
        self.protocol = StatisticalProtocol(seed=seed)
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
            
            # Convert approval_date to datetime
            self.launches['approval_date'] = pd.to_datetime(self.launches['approval_date'])
            
            print(f"Loaded {len(self.launches)} launches for backtesting")
        except FileNotFoundError:
            print("ERROR: Dataset not found. Run 'python src/cli.py build-data' first")
            sys.exit(1)
    
    def create_time_windows(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Create rolling time windows for backtesting.
        Each window represents a point in time where we make forecasts.
        """
        windows = []
        
        for year in range(self.start_year, self.end_year + 1):
            for quarter in [1, 4, 7, 10]:  # Quarterly windows
                window_date = pd.Timestamp(year=year, month=quarter, day=1)
                
                # Training data: Everything before this date
                train_end = window_date
                
                # Test data: Drugs launching in next 2 years
                test_start = window_date
                test_end = window_date + pd.DateOffset(years=2)
                
                windows.append((train_end, test_end))
        
        return windows
    
    def get_training_data(self, cutoff_date: pd.Timestamp) -> pd.DataFrame:
        """Get all drugs approved before cutoff date."""
        train_drugs = self.launches[
            self.launches['approval_date'] < cutoff_date
        ].copy()
        
        # Only include drugs with at least 2 years of data
        train_ids = []
        for drug_id in train_drugs['launch_id']:
            drug_revenues = self.revenues[self.revenues['launch_id'] == drug_id]
            if len(drug_revenues) >= 2:
                train_ids.append(drug_id)
        
        return train_drugs[train_drugs['launch_id'].isin(train_ids)]
    
    def get_test_data(self, start_date: pd.Timestamp, 
                     end_date: pd.Timestamp) -> pd.DataFrame:
        """Get drugs launching in test window."""
        test_drugs = self.launches[
            (self.launches['approval_date'] >= start_date) &
            (self.launches['approval_date'] < end_date)
        ].copy()
        
        return test_drugs
    
    def get_actuals(self, drug_ids: List[str], 
                   forecast_date: pd.Timestamp) -> Dict[str, Dict[str, float]]:
        """
        Get actual revenues for drugs at various time points.
        """
        actuals = {}
        
        for drug_id in drug_ids:
            drug_revenues = self.revenues[self.revenues['launch_id'] == drug_id]
            
            if len(drug_revenues) > 0:
                # Year 2 revenue
                y2_data = drug_revenues[drug_revenues['year_since_launch'] == 1]
                y2_actual = y2_data['revenue_usd'].values[0] if len(y2_data) > 0 else np.nan
                
                # Peak revenue (within 5 years)
                peak_data = drug_revenues[drug_revenues['year_since_launch'] <= 4]
                peak_actual = peak_data['revenue_usd'].max() if len(peak_data) > 0 else np.nan
                
                # 5-year trajectory
                trajectory = []
                for year in range(5):
                    year_data = drug_revenues[drug_revenues['year_since_launch'] == year]
                    if len(year_data) > 0:
                        trajectory.append(year_data['revenue_usd'].values[0])
                    else:
                        trajectory.append(np.nan)
                
                actuals[drug_id] = {
                    'y2': y2_actual,
                    'peak': peak_actual,
                    'trajectory': trajectory
                }
            else:
                actuals[drug_id] = {
                    'y2': np.nan,
                    'peak': np.nan,
                    'trajectory': [np.nan] * 5
                }
        
        return actuals
    
    def forecast_drugs(self, train_data: pd.DataFrame,
                      test_drugs: pd.DataFrame,
                      method: str = 'ensemble') -> Dict[str, np.ndarray]:
        """
        Generate forecasts for test drugs using specified method.
        """
        forecasts = {}
        
        if method == 'baseline':
            # Simple baseline ensemble
            for _, drug in test_drugs.iterrows():
                result = ensemble_baseline(drug, years=5)
                forecasts[drug['launch_id']] = result['ensemble']
        
        elif method == 'analog':
            # Analog-based forecast
            forecaster = AnalogForecaster()
            forecaster.launches = train_data
            forecaster.revenues = self.revenues[
                self.revenues['launch_id'].isin(train_data['launch_id'])
            ]
            forecaster.analogs = self.analogs[
                self.analogs['launch_id'].isin(train_data['launch_id']) &
                self.analogs['analog_launch_id'].isin(train_data['launch_id'])
            ]
            
            for _, drug in test_drugs.iterrows():
                forecast = forecaster.forecast_from_analogs(drug, years=5)
                forecasts[drug['launch_id']] = forecast
        
        elif method == 'patient_flow':
            # Patient flow model
            model = PatientFlowModel()
            for _, drug in test_drugs.iterrows():
                forecast = model.forecast(drug, years=5)
                forecasts[drug['launch_id']] = forecast
        
        elif method == 'ensemble':
            # Combine all methods
            for _, drug in test_drugs.iterrows():
                # Get all forecasts
                baseline_fc = ensemble_baseline(drug, years=5)['ensemble']
                
                # Analog forecast
                analog_fc = np.zeros(5)
                try:
                    forecaster = AnalogForecaster()
                    forecaster.launches = train_data
                    analog_fc = forecaster.forecast_from_analogs(drug, years=5)
                except:
                    analog_fc = baseline_fc  # Fallback
                
                # Patient flow
                flow_fc = PatientFlowModel().forecast(drug, years=5)
                
                # Weighted average
                ensemble = (baseline_fc + analog_fc + flow_fc) / 3
                forecasts[drug['launch_id']] = ensemble
        
        return forecasts
    
    def evaluate_window(self, train_end: pd.Timestamp,
                       test_end: pd.Timestamp,
                       method: str = 'ensemble') -> Dict[str, Any]:
        """
        Evaluate forecasting performance for one time window.
        """
        # Get data splits
        train_data = self.get_training_data(train_end)
        test_data = self.get_test_data(train_end, test_end)
        
        if len(test_data) == 0:
            return {'error': 'No test drugs in window'}
        
        # Generate forecasts
        forecasts = self.forecast_drugs(train_data, test_data, method)
        
        # Get actuals
        actuals = self.get_actuals(test_data['launch_id'].tolist(), test_end)
        
        # Calculate metrics
        y2_errors = []
        peak_errors = []
        trajectory_mapes = []
        
        for drug_id in forecasts:
            if drug_id in actuals:
                fc = forecasts[drug_id]
                ac = actuals[drug_id]
                
                # Y2 APE
                if not np.isnan(ac['y2']) and len(fc) > 1:
                    y2_ape = self.metrics.year2_ape(ac['y2'], fc[1])
                    y2_errors.append(y2_ape)
                
                # Peak APE
                if not np.isnan(ac['peak']):
                    peak_ape = self.metrics.peak_ape(ac['peak'], np.max(fc))
                    peak_errors.append(peak_ape)
                
                # Trajectory MAPE
                valid_points = [(a, f) for a, f in zip(ac['trajectory'], fc) 
                               if not np.isnan(a)]
                if valid_points:
                    actual_traj = np.array([a for a, _ in valid_points])
                    forecast_traj = np.array([f for _, f in valid_points])
                    mape = self.metrics.mape(actual_traj, forecast_traj)
                    trajectory_mapes.append(mape)
        
        return {
            'window_start': str(train_end),
            'window_end': str(test_end),
            'n_train': len(train_data),
            'n_test': len(test_data),
            'n_evaluated': len(y2_errors),
            'y2_ape': np.mean(y2_errors) if y2_errors else np.nan,
            'y2_ape_std': np.std(y2_errors) if y2_errors else np.nan,
            'peak_ape': np.mean(peak_errors) if peak_errors else np.nan,
            'peak_ape_std': np.std(peak_errors) if peak_errors else np.nan,
            'trajectory_mape': np.mean(trajectory_mapes) if trajectory_mapes else np.nan,
            'trajectory_mape_std': np.std(trajectory_mapes) if trajectory_mapes else np.nan
        }
    
    def run_backtest(self, methods: List[str] = None) -> Dict[str, Any]:
        """
        Run complete backtesting across all windows and methods.
        """
        if methods is None:
            methods = ['baseline', 'analog', 'patient_flow', 'ensemble']
        
        print("=" * 60)
        print("HISTORICAL BACKTESTING")
        print("=" * 60)
        print(f"Period: {self.start_year} - {self.end_year}")
        print(f"Methods: {', '.join(methods)}")
        
        # Create time windows
        windows = self.create_time_windows()
        print(f"Windows: {len(windows)} quarterly evaluation points")
        
        # Results storage
        all_results = {method: [] for method in methods}
        
        # Run backtesting for each window
        for i, (train_end, test_end) in enumerate(windows):
            print(f"\nWindow {i+1}/{len(windows)}: {train_end.date()} to {test_end.date()}")
            
            for method in methods:
                print(f"  Testing {method}...", end=" ")
                result = self.evaluate_window(train_end, test_end, method)
                
                if 'error' not in result:
                    all_results[method].append(result)
                    print(f"Y2 APE: {result['y2_ape']:.1f}%")
                else:
                    print(result['error'])
        
        # Aggregate results
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        
        summary = {}
        
        for method in methods:
            if all_results[method]:
                # Combine all windows
                all_y2 = [r['y2_ape'] for r in all_results[method] if not np.isnan(r['y2_ape'])]
                all_peak = [r['peak_ape'] for r in all_results[method] if not np.isnan(r['peak_ape'])]
                all_traj = [r['trajectory_mape'] for r in all_results[method] if not np.isnan(r['trajectory_mape'])]
                
                summary[method] = {
                    'n_windows': len(all_results[method]),
                    'y2_ape_mean': np.mean(all_y2) if all_y2 else np.nan,
                    'y2_ape_median': np.median(all_y2) if all_y2 else np.nan,
                    'y2_ape_std': np.std(all_y2) if all_y2 else np.nan,
                    'peak_ape_mean': np.mean(all_peak) if all_peak else np.nan,
                    'peak_ape_median': np.median(all_peak) if all_peak else np.nan,
                    'trajectory_mape_mean': np.mean(all_traj) if all_traj else np.nan,
                    'below_30_pct': np.mean([x < 30 for x in all_y2]) if all_y2 else 0,
                    'below_40_pct': np.mean([x < 40 for x in all_y2]) if all_y2 else 0
                }
                
                print(f"\n{method.upper()}:")
                print(f"  Y2 APE: {summary[method]['y2_ape_mean']:.1f}% (median: {summary[method]['y2_ape_median']:.1f}%)")
                print(f"  Peak APE: {summary[method]['peak_ape_mean']:.1f}%")
                print(f"  Trajectory MAPE: {summary[method]['trajectory_mape_mean']:.1f}%")
                print(f"  % forecasts <30% Y2 APE: {summary[method]['below_30_pct']:.1%}")
                print(f"  % forecasts <40% Y2 APE: {summary[method]['below_40_pct']:.1%}")
        
        # Check Gate G4
        print("\n" + "-" * 60)
        print("GATE G4 CHECK:")
        
        # Industry baseline is 40% Y2 APE
        if summary:
            best_method = min(summary.keys(), key=lambda x: summary[x]['y2_ape_mean'] if not np.isnan(summary[x]['y2_ape_mean']) else float('inf'))
            best_y2_ape = summary[best_method]['y2_ape_mean']
        else:
            print("[-] NO DATA: No forecasts were evaluated")
            best_method = 'none'
            best_y2_ape = np.nan
            gate_g4_passed = False
        
        if not np.isnan(best_y2_ape):
            if best_y2_ape < 30:
                print(f"[+] PASSED: Best method ({best_method}) achieves {best_y2_ape:.1f}% Y2 APE (<30%)")
                gate_g4_passed = True
            elif best_y2_ape < 40:
                print(f"[?] PARTIAL: Best method ({best_method}) achieves {best_y2_ape:.1f}% Y2 APE (<40%)")
                gate_g4_passed = False
            else:
                print(f"[-] FAILED: Best method ({best_method}) achieves {best_y2_ape:.1f}% Y2 APE (>=40%)")
                gate_g4_passed = False
        
        # Save results
        backtest_results = {
            'start_year': self.start_year,
            'end_year': self.end_year,
            'n_windows': len(windows),
            'methods': methods,
            'summary': summary,
            'detailed_results': all_results,
            'best_method': best_method,
            'best_y2_ape': best_y2_ape,
            'gate_g4_passed': gate_g4_passed
        }
        
        output_path = self.results_dir / "backtest_results.json"
        with open(output_path, 'w') as f:
            json.dump(backtest_results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")
        
        # Log to audit
        audit_run(
            experiment_name='backtesting',
            config={
                'start_year': self.start_year,
                'end_year': self.end_year,
                'methods': methods,
                'seed': self.seed
            },
            seed=self.seed,
            results=summary
        )
        
        return backtest_results


def main():
    """Run backtesting from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Historical Backtesting')
    parser.add_argument('--start-year', type=int, default=2015, help='Start year')
    parser.add_argument('--end-year', type=int, default=2020, help='End year')
    parser.add_argument('--methods', nargs='+', 
                       choices=['baseline', 'analog', 'patient_flow', 'ensemble'],
                       help='Methods to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    backtester = BacktestingFramework(
        start_year=args.start_year,
        end_year=args.end_year,
        seed=args.seed
    )
    
    results = backtester.run_backtest(methods=args.methods)
    
    # Return 0 if Gate G4 passed
    return 0 if results['gate_g4_passed'] else 1


if __name__ == "__main__":
    sys.exit(main())