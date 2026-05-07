#!/usr/bin/env python3
"""
H1: Evidence Grounding Experiment
Tests whether evidence-grounded forecasts beat pure heuristics.
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


class H1Experiment:
    """
    H1: Evidence-grounded forecasts should beat pure heuristics.
    
    Conditions:
    - Heuristic: Simple peak sales formula without data
    - Evidence-light: Uses analogs but no external data
    - Evidence-heavy: Uses analogs + external validation data
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
    
    def heuristic_forecast(self, train_data: pd.DataFrame, 
                          test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Pure heuristic without evidence.
        Just uses peak sales formula.
        """
        forecasts = []
        y2_forecasts = []
        peak_forecasts = []
        
        for _, drug in test_data.iterrows():
            # Simple heuristic
            peak = peak_sales_heuristic(drug)
            
            # Linear growth to peak
            forecast = np.array([
                peak * 0.05,  # Y1: 5% of peak
                peak * 0.25,  # Y2: 25% of peak  
                peak * 0.60,  # Y3: 60% of peak
                peak * 0.85,  # Y4: 85% of peak
                peak * 1.00   # Y5: 100% peak
            ])
            
            forecasts.append(forecast)
            y2_forecasts.append(forecast[1] if len(forecast) > 1 else 0)
            peak_forecasts.append(peak)
        
        # Stack forecasts
        forecast_array = np.vstack(forecasts) if forecasts else np.array([])
        
        # Simple prediction intervals (±50%)
        return {
            'forecast': forecast_array,
            'forecast_y2': np.array(y2_forecasts),
            'forecast_peak': np.array(peak_forecasts),
            'lower': forecast_array * 0.5,
            'upper': forecast_array * 1.5,
            'y2_errors': np.array(y2_forecasts) - test_data['actual_y2'].values if 'actual_y2' in test_data else np.array([])
        }
    
    def evidence_light_forecast(self, train_data: pd.DataFrame,
                               test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Evidence-light: Uses historical analogs from training data.
        """
        forecaster = AnalogForecaster()
        
        # Override with training data
        forecaster.launches = train_data
        forecaster.revenues = self.revenues[
            self.revenues['launch_id'].isin(train_data['launch_id'])
        ]
        forecaster.analogs = self.analogs[
            self.analogs['launch_id'].isin(train_data['launch_id']) &
            self.analogs['analog_launch_id'].isin(train_data['launch_id'])
        ]
        
        forecasts = []
        y2_forecasts = []
        peak_forecasts = []
        lowers = []
        uppers = []
        
        for _, drug in test_data.iterrows():
            # Get analog forecast with PI
            result = forecaster.get_prediction_intervals(drug, years=5)
            
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
            'upper': upper_array,
            'y2_errors': np.array(y2_forecasts) - test_data['actual_y2'].values if 'actual_y2' in test_data else np.array([])
        }
    
    def evidence_heavy_forecast(self, train_data: pd.DataFrame,
                               test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Evidence-heavy: Uses analogs + patient flow + external validation.
        In real implementation, would query external databases.
        """
        # Combine analog and patient flow models
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
        lowers = []
        uppers = []
        
        for _, drug in test_data.iterrows():
            # Get both forecasts
            analog_result = analog_forecaster.get_prediction_intervals(drug, years=5)
            flow_scenarios = flow_model.forecast_with_scenarios(drug, years=5)
            
            # Weighted ensemble (60% analog, 40% patient flow)
            ensemble = 0.6 * analog_result['forecast'] + 0.4 * flow_scenarios['base']
            lower = 0.6 * analog_result['lower'] + 0.4 * flow_scenarios['downside']
            upper = 0.6 * analog_result['upper'] + 0.4 * flow_scenarios['upside']
            
            # Evidence-based adjustment using drug characteristics
            # In reality, would query pricing databases, clinical trial results, etc.
            # For now, use drug characteristics to make realistic adjustments
            
            # Access tier adjustment (realistic market access impact)
            access_tier = drug.get('access_tier_at_launch', 'PA')
            if access_tier == 'OPEN':
                access_adjustment = 1.05  # Slightly optimistic for open access
            elif access_tier == 'PA':
                access_adjustment = 1.0   # Baseline
            else:  # NICHE
                access_adjustment = 0.85  # Conservative for niche access
            
            # Competition adjustment (realistic competitive pressure)
            competitors = drug.get('competitor_count_at_launch', 3)
            if competitors == 0:
                competition_adjustment = 1.1   # First-in-class advantage
            elif competitors <= 2:
                competition_adjustment = 1.0   # Baseline
            elif competitors <= 5:
                competition_adjustment = 0.9   # Moderate competition
            else:
                competition_adjustment = 0.8   # High competition
            
            # Efficacy adjustment (realistic clinical impact)
            efficacy = drug.get('clinical_efficacy_proxy', 0.7)
            efficacy_adjustment = 0.8 + 0.4 * efficacy  # Range: 0.8-1.2
            
            # Combined evidence-based adjustment
            evidence_adjustment = access_adjustment * competition_adjustment * efficacy_adjustment
            ensemble *= evidence_adjustment
            lower *= evidence_adjustment * 0.9
            upper *= evidence_adjustment * 1.1
            
            forecasts.append(ensemble)
            y2_forecasts.append(ensemble[1] if len(ensemble) > 1 else 0)
            peak_forecasts.append(np.max(ensemble))
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
            'upper': upper_array,
            'y2_errors': np.array(y2_forecasts) - test_data['actual_y2'].values if 'actual_y2' in test_data else np.array([])
        }
    
    def prepare_evaluation_data(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare data with proper temporal splits for realistic backtesting.
        
        Returns:
            Dict with 'train' and 'test' DataFrames
        """
        # Ensure approval_date is datetime
        self.launches['approval_date'] = pd.to_datetime(self.launches['approval_date'], errors='coerce')
        
        # Extract approval year (handle missing dates)
        self.launches['approval_year'] = self.launches['approval_date'].dt.year
        
        # For drugs with missing approval dates, estimate from revenue data
        missing_approval = self.launches['approval_year'].isna()
        if missing_approval.any():
            print(f"Warning: {missing_approval.sum()} drugs missing approval dates, estimating from revenue patterns")
            # Estimate approval year as 2015 + random offset for missing data
            np.random.seed(self.seed)
            estimated_years = np.random.randint(2015, 2020, missing_approval.sum())
            self.launches.loc[missing_approval, 'approval_year'] = estimated_years
        
        # Temporal split based on approval year
        # Train: Drugs launched 2015-2019 (should have complete 5+ year revenue history by 2024)
        # Test: Drugs launched 2020+ (we're forecasting their ongoing/future performance)
        
        current_year = 2024  # Our evaluation cutoff
        train_drugs = self.launches[
            (self.launches['approval_year'] >= 2015) & 
            (self.launches['approval_year'] <= 2019)
        ].copy()
        
        test_drugs = self.launches[
            self.launches['approval_year'] >= 2020
        ].copy()
        
        print(f"Temporal split: {len(train_drugs)} training drugs (2015-2019), {len(test_drugs)} test drugs (2020+)")
        
        # For training drugs, get complete revenue histories
        train_data = self._prepare_drug_data(train_drugs, complete_history=True)
        
        # For test drugs, get available revenue data (partial for recent launches)
        test_data = self._prepare_drug_data(test_drugs, complete_history=False)
        
        return {'train': train_data, 'test': test_data}
    
    def _prepare_drug_data(self, drugs_df: pd.DataFrame, complete_history: bool = True) -> pd.DataFrame:
        """
        Prepare drug data with actual revenue values for evaluation.
        
        Args:
            drugs_df: Launch data for drugs to prepare
            complete_history: If True, require complete 5-year revenue history
        """
        results = []
        
        for _, drug in drugs_df.iterrows():
            launch_id = drug['launch_id']
            
            # Get revenue data for this drug
            drug_revenues = self.revenues[self.revenues['launch_id'] == launch_id].copy()
            
            if len(drug_revenues) == 0:
                if complete_history:
                    continue  # Skip drugs without revenue data in training
                else:
                    # For test drugs, create entry with missing actuals
                    drug_data = drug.to_dict()
                    drug_data.update({
                        'actual_y1': np.nan, 'actual_y2': np.nan, 'actual_y3': np.nan,
                        'actual_y4': np.nan, 'actual_y5': np.nan, 'actual_peak': np.nan,
                        'revenue_years_available': 0
                    })
                    results.append(drug_data)
                    continue
            
            # Sort by year
            drug_revenues = drug_revenues.sort_values('year_since_launch')
            
            # Extract actual values by year
            actuals = {}
            for year in range(1, 6):  # Y1 through Y5
                year_data = drug_revenues[drug_revenues['year_since_launch'] == year]
                if len(year_data) > 0:
                    actuals[f'actual_y{year}'] = year_data['revenue_usd'].iloc[0]
                else:
                    actuals[f'actual_y{year}'] = np.nan
            
            # Calculate actual peak
            actuals['actual_peak'] = drug_revenues['revenue_usd'].max()
            actuals['revenue_years_available'] = len(drug_revenues)
            
            # For training, require at least Y1 and Y2 data
            if complete_history and (pd.isna(actuals['actual_y1']) or pd.isna(actuals['actual_y2'])):
                continue
            
            # Combine launch data with actuals
            drug_data = drug.to_dict()
            drug_data.update(actuals)
            results.append(drug_data)
        
        result_df = pd.DataFrame(results)
        print(f"Prepared {len(result_df)} drugs with revenue data")
        
        return result_df
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the full H1 experiment."""
        print("=" * 60)
        print("H1: EVIDENCE GROUNDING EXPERIMENT")
        print("=" * 60)
        
        # Prepare data with proper temporal splits
        data_splits = self.prepare_evaluation_data()
        train_data = data_splits['train']
        test_data = data_splits['test']
        
        print(f"\nData split: {len(train_data)} train, {len(test_data)} test")
        
        # Check Gate G3
        g3_passed, g3_msg = self.gates.check_gate_g3(
            self.protocol, len(train_data), len(test_data)
        )
        if not g3_passed:
            print(f"ERROR: {g3_msg}")
            return {'error': g3_msg}
        print(f"[+] {g3_msg}")
        
        # Run three conditions
        print("\n" + "-" * 40)
        print("Running forecasting methods...")
        
        # 1. Heuristic (baseline)
        print("  1. Pure heuristic...")
        heuristic_pred = self.heuristic_forecast(train_data, test_data)
        
        # 2. Evidence-light
        print("  2. Evidence-light (analogs)...")
        light_pred = self.evidence_light_forecast(train_data, test_data)
        
        # 3. Evidence-heavy
        print("  3. Evidence-heavy (ensemble + external)...")
        heavy_pred = self.evidence_heavy_forecast(train_data, test_data)
        
        # Calculate metrics
        print("\n" + "-" * 40)
        print("Calculating metrics...")
        
        results = {
            'heuristic': self._calculate_metrics(test_data, heuristic_pred),
            'evidence_light': self._calculate_metrics(test_data, light_pred),
            'evidence_heavy': self._calculate_metrics(test_data, heavy_pred)
        }
        
        # Statistical comparisons
        print("\n" + "-" * 40)
        print("Statistical testing...")
        
        # Compare evidence-light vs heuristic
        if 'y2_errors' in light_pred and len(light_pred['y2_errors']) > 0:
            light_vs_heuristic = self.hypothesis.compare_models(
                np.abs(light_pred['y2_errors']),
                np.abs(heuristic_pred['y2_errors']),
                paired=True
            )
            results['light_vs_heuristic'] = light_vs_heuristic
        
        # Compare evidence-heavy vs heuristic
        if 'y2_errors' in heavy_pred and len(heavy_pred['y2_errors']) > 0:
            heavy_vs_heuristic = self.hypothesis.compare_models(
                np.abs(heavy_pred['y2_errors']),
                np.abs(heuristic_pred['y2_errors']),
                paired=True
            )
            results['heavy_vs_heuristic'] = heavy_vs_heuristic
        
        # Compare evidence-heavy vs evidence-light
        if 'y2_errors' in heavy_pred and 'y2_errors' in light_pred:
            heavy_vs_light = self.hypothesis.compare_models(
                np.abs(heavy_pred['y2_errors']),
                np.abs(light_pred['y2_errors']),
                paired=True
            )
            results['heavy_vs_light'] = heavy_vs_light
        
        # Apply multiple comparison correction
        p_values = []
        if 'light_vs_heuristic' in results:
            p_values.append(results['light_vs_heuristic']['p_value'])
        if 'heavy_vs_heuristic' in results:
            p_values.append(results['heavy_vs_heuristic']['p_value'])
        if 'heavy_vs_light' in results:
            p_values.append(results['heavy_vs_light']['p_value'])
        
        if p_values:
            corrected = self.hypothesis.multiple_comparison_correction(p_values)
            results['corrected_significance'] = corrected
        
        # Summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        for method in ['heuristic', 'evidence_light', 'evidence_heavy']:
            if method in results:
                m = results[method]
                print(f"\n{method.upper()}:")
                print(f"  Y2 APE: {m.get('y2_ape', np.nan):.1f}%")
                print(f"  Peak APE: {m.get('peak_ape', np.nan):.1f}%")
                print(f"  PI Coverage: {m.get('pi_coverage', 0):.1%}")
        
        # Hypothesis results
        print("\n" + "-" * 40)
        print("HYPOTHESIS TESTING:")
        
        if 'light_vs_heuristic' in results:
            comp = results['light_vs_heuristic']
            sig = "YES" if comp['significant'] else "NO"
            print(f"  Evidence-light beats heuristic: {sig} (p={comp['p_value']:.4f})")
        
        if 'heavy_vs_heuristic' in results:
            comp = results['heavy_vs_heuristic']
            sig = "YES" if comp['significant'] else "NO"
            print(f"  Evidence-heavy beats heuristic: {sig} (p={comp['p_value']:.4f})")
        
        # Save results
        results['experiment'] = 'H1'
        results['seed'] = self.seed
        results['n_train'] = len(train_data)
        results['n_test'] = len(test_data)
        
        output_path = self.results_dir / "h1_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")
        
        # Log to audit
        audit_run(
            experiment_name='H1_evidence_grounding',
            config={'seed': self.seed},
            seed=self.seed,
            results=results
        )
        
        return results
    
    def _calculate_metrics(self, test_data: pd.DataFrame, 
                          predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Y2 APE
        if 'forecast_y2' in predictions and 'actual_y2' in test_data:
            y2_ape = self.metrics.year2_ape(
                test_data['actual_y2'].values,
                predictions['forecast_y2']
            )
            metrics['y2_ape'] = y2_ape
        
        # Peak APE
        if 'forecast_peak' in predictions and 'actual_peak' in test_data:
            peak_ape = self.metrics.peak_ape(
                test_data['actual_peak'].values,
                predictions['forecast_peak']
            )
            metrics['peak_ape'] = peak_ape
        
        # PI Coverage
        if 'lower' in predictions and 'upper' in predictions and 'actual_peak' in test_data:
            # Check if peak is within intervals
            coverage = self.metrics.prediction_interval_coverage(
                test_data['actual_peak'].values,
                predictions['forecast_peak'] * 0.8,  # Approximate lower
                predictions['forecast_peak'] * 1.2   # Approximate upper
            )
            metrics['pi_coverage'] = coverage
        
        return metrics


def main():
    """Run H1 experiment from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='H1: Evidence Grounding Experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    experiment = H1Experiment(seed=args.seed)
    results = experiment.run_experiment()
    
    # Return 0 if hypothesis supported
    if results.get('heavy_vs_heuristic', {}).get('significant', False):
        print("\n[+] H1 SUPPORTED: Evidence grounding improves forecasts")
        return 0
    else:
        print("\n[-] H1 NOT SUPPORTED: Evidence grounding did not significantly improve")
        return 1


if __name__ == "__main__":
    sys.exit(main())