#!/usr/bin/env python3
"""
Phase 5: Real Historical Validation with Full Multi-Agent System
Following Linus principles: Test the actual system, not simulations
"""

import pandas as pd
import numpy as np
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime
import time

# Add paths
ai_scientist_path = str(Path(__file__).parent.parent / "ai_scientist")
src_path = str(Path(__file__).parent.parent / "src")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from gpt5_orchestrator import GPT5Orchestrator
from models.baselines import peak_sales_heuristic, ensemble_baseline
from models.analogs import AnalogForecaster
from models.ta_priors import get_ta_peak_share_prior, get_ta_y2_share_prior
from system_monitor import reset_system_monitor, get_system_monitor

class RealHistoricalValidator:
    """
    Phase 5: Test our actual multi-agent system against real historical outcomes
    Uses real LLM calls (GPT-5, DeepSeek, Perplexity, Claude) not simulations
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.timings = {}
        self._log_timing("script_start", "Phase 5 validation script started")
        
        self._log_timing("data_load_start", "Starting data loading")
        self.launches = pd.read_parquet('data_proc/launches.parquet')
        self._log_timing("launches_loaded", f"Loaded {len(self.launches)} launches")
        
        # Use fixed revenue data if available, otherwise original
        fixed_revenue_path = Path('data_proc/launch_revenues_fixed.parquet')
        if fixed_revenue_path.exists():
            self.revenues = pd.read_parquet('data_proc/launch_revenues_fixed.parquet')
            self._log_timing("revenue_data_loaded", "Using FIXED revenue data (post-SEC extraction fix)")
        else:
            self.revenues = pd.read_parquet('data_proc/launch_revenues.parquet')
            self._log_timing("revenue_data_loaded", "Using ORIGINAL revenue data (may have SEC extraction issues)")
        
        # MUST_GET_RIGHT cases from MASSIVE_OVERHAUL_PLAN.md
        self.must_get_right = {
            'blockbusters': ['Keytruda', 'Humira', 'Eliquis'],  # Must predict massive success
            'failures': ['Exubera', 'Belsomra', 'Addyi'],       # Must predict failure  
            'moderate': ['Repatha', 'Entresto', 'Aimovig']      # Must predict access challenges
        }
        
        # Industry consultant baseline (±40% MAPE per plan)
        self.consultant_mape_target = 0.40
        self.our_mape_target = 0.25  # Our target: 25% MAPE
        
        self._log_timing("init_complete", f"Initialization complete: {len(self.launches)} launches, {len(self.revenues)} revenue records")
    
    def _log_timing(self, step: str, description: str):
        """Log timing for performance analysis"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.timings[step] = {
            'elapsed_total': elapsed,
            'timestamp': current_time,
            'description': description
        }
        print(f"[{elapsed:6.2f}s] {description}")
    
    def _get_step_duration(self, step1: str, step2: str) -> float:
        """Get duration between two timing steps"""
        if step1 in self.timings and step2 in self.timings:
            return self.timings[step2]['elapsed_total'] - self.timings[step1]['elapsed_total']
        return 0.0
    
    def select_validation_drugs(self) -> List[Dict[str, Any]]:
        """Select key drugs for full multi-agent validation"""
        
        self._log_timing("drug_selection_start", "Starting drug selection")
        
        # Find drugs we have both launch data and revenue data for
        drugs_with_revenue = set(self.revenues['launch_id'].unique())
        available_drugs = self.launches[self.launches['launch_id'].isin(drugs_with_revenue)]
        self._log_timing("drug_filtering", f"Found {len(available_drugs)} drugs with both launch and revenue data")
        
        validation_drugs = []
        
        # Priority 1: MUST_GET_RIGHT drugs that we have data for
        all_must_get_right = []
        for category, drugs in self.must_get_right.items():
            all_must_get_right.extend(drugs)
        
        priority_drugs = available_drugs[available_drugs['drug_name'].isin(all_must_get_right)]
        
        for _, drug in priority_drugs.iterrows():
            launch_id = drug['launch_id']
            actual_revenues = self.get_actual_revenues(launch_id)
            
            if np.max(actual_revenues) > 0:  # Has real revenue data
                validation_drugs.append({
                    'drug': drug,
                    'actual_revenues': actual_revenues,
                    'category': self.get_drug_category(drug['drug_name']),
                    'priority': 'must_get_right'
                })
        
        # Priority 2: Add a few more representative drugs
        other_drugs = available_drugs[~available_drugs['drug_name'].isin(all_must_get_right)]
        for _, drug in other_drugs.head(3).iterrows():
            launch_id = drug['launch_id']
            actual_revenues = self.get_actual_revenues(launch_id)
            
            if np.max(actual_revenues) > 0:
                validation_drugs.append({
                    'drug': drug,
                    'actual_revenues': actual_revenues,
                    'category': 'representative',
                    'priority': 'additional'
                })
        
        self._log_timing("drug_selection_complete", f"Selected {len(validation_drugs)} drugs for validation")
        
        for vd in validation_drugs:
            drug_name = vd['drug']['drug_name']
            actual_peak = np.max(vd['actual_revenues'])
            print(f"  - {drug_name}: ${actual_peak:,.0f} peak ({vd['category']})")
        
        return validation_drugs
    
    def get_drug_category(self, drug_name: str) -> str:
        """Determine if drug is blockbuster, failure, or moderate"""
        for category, drugs in self.must_get_right.items():
            if drug_name in drugs:
                return category
        return 'other'
    
    def get_actual_revenues(self, launch_id: str, years: int = 5) -> np.ndarray:
        """Get actual revenue trajectory for validation"""
        
        drug_revenues = self.revenues[self.revenues['launch_id'] == launch_id]
        drug_revenues = drug_revenues.sort_values('year_since_launch')
        
        actual = np.zeros(years)
        for _, row in drug_revenues.iterrows():
            year = int(row['year_since_launch'])
            if 0 <= year < years:
                actual[year] = row['revenue_usd']
        
        return actual
    
    async def run_multi_agent_forecast(self, drug: pd.Series) -> Dict[str, Any]:
        """Run our actual multi-agent system on a historical drug"""
        
        drug_name = drug['drug_name']
        multiagent_start = time.time()
        print(f"\n--- Running Multi-Agent System: {drug_name} ---")
        
        # Reset monitor for this drug
        reset_system_monitor()
        
        try:
            # Initialize our actual GPT-5 orchestrator
            init_start = time.time()
            orchestrator = GPT5Orchestrator()
            init_time = time.time() - init_start
            print(f"[+{init_time:5.2f}s] GPT-5 Orchestrator initialized")
            
            # Create a realistic query as if we were forecasting at launch
            therapeutic_area = drug.get('therapeutic_area', 'Unknown')
            indication = drug.get('indication', 'Unknown')
            
            query = f"Commercial forecast for {drug_name} in {therapeutic_area} for {indication}"
            
            print(f"Query: {query}")
            print("Executing full 8-step multi-agent pipeline...")
            print("  Step 1: Query parsing (GPT-5)...")
            print("  Step 2: Data collection (DeepSeek)...")
            print("  Step 3: Data review (Perplexity)...")
            print("  Step 4: Market analysis (GPT-5)...")
            print("  Step 5: Multi-method forecasting...")
            print("  Step 6: Harsh review (Perplexity)...")
            print("  [Progress updates will appear as pipeline executes]")
            
            # Run the actual multi-agent system
            pipeline_start = time.time()
            start_time = datetime.now()
            result = await orchestrator.process_drug_forecast(query)
            end_time = datetime.now()
            pipeline_time = time.time() - pipeline_start
            
            execution_time = (end_time - start_time).total_seconds()
            print(f"[+{pipeline_time:5.2f}s] Multi-agent pipeline completed in {execution_time:.1f} seconds")
            
            # Extract forecast from result
            extract_start = time.time()
            forecast_data = result.get('forecast', {})
            confidence = result.get('confidence', 0)
            extract_time = time.time() - extract_start
            print(f"[+{extract_time:5.2f}s] Forecast data extracted")
            
            # Convert to 5-year revenue trajectory
            # Extract from our orchestrator's actual output format
            if isinstance(forecast_data, dict):
                peak_sales = forecast_data.get('peak_sales_forecast', 0)
                print(f"DEBUG: forecast_data keys: {list(forecast_data.keys())}")
                print(f"DEBUG: extracted peak_sales: ${peak_sales:,.0f}")
                
                if peak_sales > 0:
                    # Create realistic trajectory
                    forecast_trajectory = np.array([
                        peak_sales * 0.1,   # Y0: Launch year
                        peak_sales * 0.4,   # Y1: Building
                        peak_sales * 0.8,   # Y2: Growing  
                        peak_sales * 1.0,   # Y3: Peak
                        peak_sales * 0.9    # Y4: Mature
                    ])
                else:
                    forecast_trajectory = np.zeros(5)
            else:
                print(f"DEBUG: forecast_data is not dict, type: {type(forecast_data)}")
                forecast_trajectory = np.zeros(5)
            
            # Get monitoring data and usage from audit trail
            monitor = get_system_monitor()
            audit_trail = monitor.generate_audit_trail()
            
            # Extract actual costs and tokens from audit trail
            execution_summary = audit_trail.get('execution_summary', {})
            actual_cost = execution_summary.get('total_cost_usd', 0.0)
            actual_tokens = execution_summary.get('total_tokens', 0)
            
            return {
                'forecast_trajectory': forecast_trajectory,
                'confidence': confidence,
                'execution_time': execution_time,
                'total_cost': actual_cost,
                'total_tokens': actual_tokens,
                'decisions_made': len(monitor.decisions),
                'audit_trail': audit_trail,
                'raw_result': result
            }
            
        except Exception as e:
            print(f"Multi-agent system failed for {drug_name}: {e}")
            return {
                'forecast_trajectory': np.zeros(5),
                'confidence': 0,
                'execution_time': 0,
                'total_cost': 0,
                'total_tokens': 0,
                'decisions_made': 0,
                'error': str(e)
            }
    
    def run_baseline_forecasts(self, drug: pd.Series) -> Dict[str, np.ndarray]:
        """Run industry baseline methods for comparison"""
        
        drug_name = drug['drug_name']
        print(f"Running baseline methods for {drug_name}...")
        
        forecasts = {}
        
        # Peak Sales Heuristic (consultant method)
        try:
            peak_forecast = peak_sales_heuristic(drug)
            if peak_forecast > 0:
                # Convert to trajectory
                forecasts['peak_heuristic'] = np.array([
                    peak_forecast * 0.2, peak_forecast * 0.6, peak_forecast * 1.0,
                    peak_forecast * 0.9, peak_forecast * 0.8
                ])
            else:
                forecasts['peak_heuristic'] = np.zeros(5)
        except Exception as e:
            print(f"Peak heuristic failed: {e}")
            forecasts['peak_heuristic'] = np.zeros(5)
        
        # Ensemble Baseline
        try:
            ensemble_result = ensemble_baseline(drug, years=5)
            if isinstance(ensemble_result, dict) and 'ensemble' in ensemble_result:
                forecasts['ensemble_baseline'] = ensemble_result['ensemble']
            else:
                forecasts['ensemble_baseline'] = np.zeros(5)
        except Exception as e:
            print(f"Ensemble baseline failed: {e}")
            forecasts['ensemble_baseline'] = np.zeros(5)
        
        # Enhanced Analog Forecasting (our latest improvement)
        try:
            print(f"Running enhanced analog forecasting for {drug_name}...")
            analog_forecaster = AnalogForecaster()
            analog_forecast = analog_forecaster.forecast_from_analogs(drug, years=5)
            if isinstance(analog_forecast, np.ndarray) and len(analog_forecast) == 5:
                forecasts['analog_enhanced'] = analog_forecast
                print(f"Analog forecast peak: ${np.max(analog_forecast):,.0f}")
            else:
                forecasts['analog_enhanced'] = np.zeros(5)
        except Exception as e:
            print(f"Enhanced analog forecasting failed: {e}")
            forecasts['analog_enhanced'] = np.zeros(5)
        
        return forecasts
    
    def calculate_accuracy_metrics(self, forecast: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
        """Calculate industry-standard accuracy metrics (fixed per GPT-5 feedback)"""
        
        if len(forecast) == 0 or len(actual) == 0:
            return {'mape': 999.0, 'peak_ape': 999.0, 'y2_ape': 999.0}
        
        # MAPE (Mean Absolute Percentage Error)
        valid_errors = []
        for i in range(min(len(forecast), len(actual))):
            if actual[i] > 0:
                error = abs(forecast[i] - actual[i]) / actual[i]
                valid_errors.append(error)
        
        mape = np.mean(valid_errors) if valid_errors else 999.0
        
        # Peak APE (Absolute Percentage Error) - GPT-5 correction
        actual_peak = np.max(actual)
        forecast_peak = np.max(forecast)
        peak_ape = abs(forecast_peak - actual_peak) / actual_peak if actual_peak > 0 else 999.0
        
        # Year 2 APE (Absolute Percentage Error) - GPT-5 correction
        if len(actual) > 1 and actual[1] > 0:
            y2_ape = abs(forecast[1] - actual[1]) / actual[1]
        else:
            y2_ape = 999.0
        
        return {
            'mape': mape,
            'peak_ape': peak_ape,    # GPT-5 fix: renamed from peak_accuracy
            'y2_ape': y2_ape         # GPT-5 fix: renamed from y2_accuracy
        }
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete Phase 5 validation with real multi-agent system"""
        
        self._log_timing("validation_start", "Starting Phase 5 full validation")
        
        print("\n" + "="*80)
        print("PHASE 5: REAL HISTORICAL VALIDATION")
        print("Full Multi-Agent System vs Industry Baselines")
        print("="*80)
        
        # Select drugs for validation (limit to 2 for quick test)
        validation_drugs = self.select_validation_drugs()[:2]  # Quick test with 2 drugs
        self._log_timing("drugs_selected", f"Selected {len(validation_drugs)} drugs for validation")
        
        if not validation_drugs:
            return {'error': 'No suitable drugs found for validation'}
        
        results = []
        total_cost = 0
        total_execution_time = 0
        
        for i, vd in enumerate(validation_drugs):
            drug = vd['drug']
            actual_revenues = vd['actual_revenues']
            drug_name = drug['drug_name']
            
            self._log_timing(f"drug_{i+1}_start", f"Starting validation for drug {i+1}/{len(validation_drugs)}: {drug_name}")
            
            print(f"\n{'='*60}")
            print(f"VALIDATION {i+1}/{len(validation_drugs)}: {drug_name}")
            print(f"Category: {vd['category']}")
            print(f"Actual peak revenue: ${np.max(actual_revenues):,.0f}")
            print(f"{'='*60}")
            
            # Run our multi-agent system
            self._log_timing(f"drug_{i+1}_multiagent_start", f"Starting multi-agent forecast for {drug_name}")
            multi_agent_result = await self.run_multi_agent_forecast(drug)
            multi_agent_forecast = multi_agent_result['forecast_trajectory']
            self._log_timing(f"drug_{i+1}_multiagent_complete", f"Multi-agent forecast complete for {drug_name} ({multi_agent_result['execution_time']:.1f}s)")
            
            # Run baseline methods
            self._log_timing(f"drug_{i+1}_baselines_start", f"Running baseline forecasts for {drug_name}")
            baseline_forecasts = self.run_baseline_forecasts(drug)
            self._log_timing(f"drug_{i+1}_baselines_complete", f"Baseline forecasts complete for {drug_name}")
            
            # Calculate metrics for all methods
            drug_result = {
                'drug_name': drug_name,
                'therapeutic_area': drug.get('therapeutic_area', 'Unknown'),
                'category': vd['category'],
                'actual_revenues': actual_revenues.tolist(),
                'actual_peak': float(np.max(actual_revenues)),
                'forecasts': {},
                'metrics': {},
                'multi_agent_meta': {
                    'execution_time': multi_agent_result['execution_time'],
                    'cost': multi_agent_result['total_cost'],
                    'tokens': multi_agent_result['total_tokens'],
                    'decisions': multi_agent_result['decisions_made'],
                    'confidence': multi_agent_result['confidence']
                }
            }
            
            # Multi-agent results
            multi_metrics = self.calculate_accuracy_metrics(multi_agent_forecast, actual_revenues)
            drug_result['forecasts']['multi_agent'] = multi_agent_forecast.tolist()
            drug_result['metrics']['multi_agent'] = multi_metrics
            
            print(f"\nMULTI-AGENT SYSTEM:")
            print(f"  Peak forecast: ${np.max(multi_agent_forecast):,.0f}")
            print(f"  MAPE: {multi_metrics['mape']:.1%}")
            print(f"  Peak APE: {multi_metrics['peak_ape']:.1%}")
            print(f"  Cost: ${multi_agent_result['total_cost']:.4f}")
            print(f"  Tokens: {multi_agent_result['total_tokens']:,}")
            
            # Baseline results
            for method_name, forecast in baseline_forecasts.items():
                metrics = self.calculate_accuracy_metrics(forecast, actual_revenues)
                drug_result['forecasts'][method_name] = forecast.tolist()
                drug_result['metrics'][method_name] = metrics
                
                print(f"\n{method_name.upper()}:")
                print(f"  Peak forecast: ${np.max(forecast):,.0f}")
                print(f"  MAPE: {metrics['mape']:.1%}")
                print(f"  Peak APE: {metrics['peak_ape']:.1%}")
            
            results.append(drug_result)
            total_cost += multi_agent_result['total_cost']
            total_execution_time += multi_agent_result['execution_time']
        
        # Summarize results
        summary = self.analyze_validation_results(results, total_cost, total_execution_time)
        
        # Save results (convert numpy types for JSON serialization)
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        # Deep convert results
        json_safe_results = json.loads(json.dumps(results, default=convert_for_json))
        json_safe_summary = json.loads(json.dumps(summary, default=convert_for_json))
        
        with open('results/phase5_real_validation.json', 'w') as f:
            json.dump({
                'summary': json_safe_summary,
                'detailed_results': json_safe_results,
                'validation_timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\nValidation results saved to: results/phase5_real_validation.json")
        
        return summary
    
    def analyze_validation_results(self, results: List[Dict], total_cost: float, 
                                 total_execution_time: float) -> Dict[str, Any]:
        """Analyze and summarize validation results"""
        
        print(f"\n" + "="*80)
        print("PHASE 5 VALIDATION ANALYSIS")
        print("="*80)
        
        # Collect metrics by method  
        methods = ['multi_agent', 'peak_heuristic', 'ensemble_baseline', 'analog_enhanced']
        method_metrics = {method: [] for method in methods}
        
        for result in results:
            for method in methods:
                if method in result['metrics']:
                    metrics = result['metrics'][method]
                    if metrics['mape'] < 5.0:  # Exclude outliers
                        method_metrics[method].append(metrics['mape'])
        
        # Calculate average performance
        summary = {
            'total_drugs_tested': len(results),
            'total_cost': total_cost,
            'total_execution_time': total_execution_time,
            'method_performance': {},
            'phase5_success': False
        }
        
        for method in methods:
            if method_metrics[method]:
                avg_mape = np.mean(method_metrics[method])
                summary['method_performance'][method] = {
                    'avg_mape': avg_mape,
                    'valid_predictions': len(method_metrics[method])
                }
                
                print(f"\n{method.upper()}: {avg_mape:.1%} average MAPE")
        
        # Check Phase 5 success criteria
        if 'multi_agent' in summary['method_performance']:
            multi_mape = summary['method_performance']['multi_agent']['avg_mape']
            
            success_criteria = [
                multi_mape < self.consultant_mape_target,  # Beat consultant baseline
                len(results) >= 3,  # Tested on multiple drugs
                total_cost < 50.0    # Reasonable cost
            ]
            
            summary['phase5_success'] = all(success_criteria)
            summary['beat_consultant_baseline'] = multi_mape < self.consultant_mape_target
            summary['achieved_target_accuracy'] = multi_mape < self.our_mape_target
            
            print(f"\nPHASE 5 SUCCESS ANALYSIS:")
            print(f"  Multi-agent MAPE: {multi_mape:.1%}")
            print(f"  Consultant baseline: {self.consultant_mape_target:.1%}")
            print(f"  Beat consultant: {summary['beat_consultant_baseline']}")
            print(f"  Our target (25%): {summary['achieved_target_accuracy']}")
            print(f"  Total cost: ${total_cost:.2f}")
            print(f"  Phase 5 success: {summary['phase5_success']}")
        
        return summary

async def main():
    """Main Phase 5 real validation with comprehensive timing"""
    
    script_start = time.time()
    print(f"[{0:6.2f}s] Phase 5 validation script execution started")
    
    try:
        validator = RealHistoricalValidator()
        init_time = time.time() - script_start
        print(f"[{init_time:6.2f}s] Validator initialization complete")
        
        results = await validator.run_full_validation()
        
        validation_time = time.time() - script_start
        print(f"[{validation_time:6.2f}s] Full validation complete")
        
        # Print timing summary
        print(f"\n" + "="*80)
        print("TIMING SUMMARY:")
        for step, timing in validator.timings.items():
            print(f"  {step:25}: {timing['elapsed_total']:6.2f}s - {timing['description']}")
        print("="*80)
        
        print(f"\n" + "="*80)
        if results.get('phase5_success', False):
            print("PHASE 5 REAL VALIDATION: SUCCESS")
            print("Multi-agent system validated against historical outcomes")
            print("Ready to proceed to Phase 6: Paper completion")
        else:
            print("PHASE 5 REAL VALIDATION: NEEDS IMPROVEMENT")
            print("System performance below targets on historical data")
        print("="*80)
        
        return results.get('phase5_success', False)
        
    except Exception as e:
        error_time = time.time() - script_start
        print(f"[{error_time:6.2f}s] PHASE 5 VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(main())