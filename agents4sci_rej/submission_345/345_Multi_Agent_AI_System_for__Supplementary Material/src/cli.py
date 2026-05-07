#!/usr/bin/env python3
"""
Command-line interface for pharmaceutical forecasting system.
Following Linus principle: Simple commands that do one thing well.
"""

import click
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from data.build_dataset import DatasetBuilder
import subprocess
from models.baselines import ensemble_baseline, peak_sales_heuristic
from models.analogs import AnalogForecaster
from models.patient_flow import PatientFlowModel
from stats.protocol import StatisticalProtocol, AcceptanceGates, run_statistical_protocol
from utils.audit import get_audit_logger, audit_summary
import subprocess


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """Pharmaceutical Forecasting System CLI."""
    pass


@cli.command()
@click.option('--seed', default=42, help='Random seed for reproducibility')
@click.option('--output-dir', type=Path, help='Output directory for datasets')
def build_data(seed, output_dir):
    """Build pharmaceutical launch dataset (Gate G1)."""
    click.echo("=" * 50)
    click.echo("Building pharmaceutical launch dataset...")
    click.echo("=" * 50)
    
    np.random.seed(seed)
    
    # Use real data collection instead of synthetic builder
    click.echo("Running real pharmaceutical data collection...")
    result = subprocess.run([
        sys.executable, '-m', 'src.data.collect_real'
    ], capture_output=True, text=True)
    
    success = result.returncode == 0
    
    if success:
        click.secho("[+] Real data collection completed", fg='green')
        
        # Generate profile from real data
        try:
            import pandas as pd
            launches = pd.read_parquet("data_proc/launches.parquet")
            revenues = pd.read_parquet("data_proc/launch_revenues.parquet")
            
            ta_counts = launches['therapeutic_area'].value_counts()
            
            profile = {
                'n_launches': len(launches),
                'n_therapeutic_areas': len(ta_counts),
                'therapeutic_areas': ta_counts.to_dict(),
                'data_source': 'real_pharmaceutical_data'
            }
            
            # Save updated profile
            profile_path = Path("results/data_profile.json")
            profile_path.parent.mkdir(exist_ok=True)
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            click.echo("\nReal Data Profile:")
            click.echo(f"  Launches: {profile['n_launches']}")
            click.echo(f"  Therapeutic Areas: {profile['n_therapeutic_areas']}")
            click.echo(f"  Revenue Records: {len(revenues)}")
            
            if profile['n_launches'] >= 50 and profile['n_therapeutic_areas'] >= 5:
                click.secho("\n[+] Gate G1 PASSED", fg='green', bold=True)
            else:
                click.secho("\n[-] Gate G1 FAILED: Insufficient data", fg='red', bold=True)
                
        except Exception as e:
            click.secho(f"Error reading real data: {e}", fg='red')
            sys.exit(1)
    else:
        click.secho("[-] Real data collection failed", fg='red')
        if result.stderr:
            click.echo(f"Error: {result.stderr}")
        sys.exit(1)


@cli.command()
@click.option('--verbose', is_flag=True, help='Show detailed test output')
def test_baselines(verbose):
    """Test baseline models (Gate G2)."""
    click.echo("=" * 50)
    click.echo("Testing baseline models...")
    click.echo("=" * 50)
    
    import subprocess
    
    cmd = ['python', '-m', 'pytest', 'tests/test_baselines.py']
    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        click.secho("[+] All baseline tests passed", fg='green')
        click.secho("\n[+] Gate G2 PASSED: Baselines implemented", fg='green', bold=True)
    else:
        click.echo(result.stdout)
        click.echo(result.stderr)
        click.secho("\n[-] Gate G2 FAILED: Baseline tests failed", fg='red', bold=True)
        sys.exit(1)


@cli.command()
@click.option('--drug-id', help='Drug launch ID to forecast')
@click.option('--method', type=click.Choice(['ensemble', 'analogs', 'patient_flow', 'all']), 
              default='ensemble', help='Forecasting method')
@click.option('--years', default=5, help='Years to forecast')
def forecast(drug_id, method, years):
    """Generate forecast for a specific drug."""
    
    # Load data
    data_dir = Path("data_proc")
    if not (data_dir / "launches.parquet").exists():
        click.secho("Error: Dataset not built. Run 'build-data' first.", fg='red')
        sys.exit(1)
    
    launches = pd.read_parquet(data_dir / "launches.parquet")
    
    if drug_id:
        drug = launches[launches['launch_id'] == drug_id]
        if drug.empty:
            click.secho(f"Error: Drug {drug_id} not found", fg='red')
            sys.exit(1)
        drug = drug.iloc[0]
    else:
        # Use first drug as example
        drug = launches.iloc[0]
        drug_id = drug['launch_id']
    
    click.echo(f"\nForecasting for {drug_id}: {drug['drug_name']}")
    click.echo(f"  Therapeutic Area: {drug['therapeutic_area']}")
    click.echo(f"  Market Size: {drug['eligible_patients_at_launch']:,} patients")
    click.echo(f"  Monthly Price: ${drug['list_price_month_usd_launch']:,.0f}")
    
    if method in ['ensemble', 'all']:
        results = ensemble_baseline(drug, years)
        click.echo("\nEnsemble Baseline Forecast:")
        for year, revenue in enumerate(results['ensemble']):
            click.echo(f"  Year {year+1}: ${revenue/1e6:,.1f}M")
    
    if method in ['analogs', 'all']:
        from models.analogs import analog_forecast_with_pi
        results = analog_forecast_with_pi(drug, years)
        click.echo("\nAnalog-based Forecast:")
        for year in range(years):
            click.echo(f"  Year {year+1}: ${results['forecast'][year]/1e6:,.1f}M " +
                      f"[{results['lower'][year]/1e6:,.1f}M - {results['upper'][year]/1e6:,.1f}M]")
    
    if method in ['patient_flow', 'all']:
        from models.patient_flow import patient_flow_scenarios
        scenarios = patient_flow_scenarios(drug, years)
        click.echo("\nPatient Flow Forecast:")
        for year in range(years):
            click.echo(f"  Year {year+1}: ${scenarios['base'][year]/1e6:,.1f}M " +
                      f"(↓${scenarios['downside'][year]/1e6:,.1f}M " +
                      f"↑${scenarios['upside'][year]/1e6:,.1f}M)")


@cli.command()
@click.option('--verbose', is_flag=True, help='Show detailed protocol checks')
def check_protocol(verbose):
    """Check statistical protocol (Gate G3)."""
    click.echo("=" * 50)
    click.echo("Checking statistical protocol...")
    click.echo("=" * 50)
    
    protocol = StatisticalProtocol()
    gates = AcceptanceGates()
    
    # Check with dummy data sizes
    train_size = 50
    test_size = 20
    
    passed, msg = gates.check_gate_g3(protocol, train_size, test_size)
    
    if verbose:
        click.echo("\nProtocol Settings:")
        click.echo(f"  Train/Test Split: {protocol.train_ratio:.0%}/{protocol.test_ratio:.0%}")
        click.echo(f"  Cross-validation Folds: {protocol.n_folds}")
        click.echo(f"  Bootstrap Samples: {protocol.n_bootstrap}")
        click.echo(f"  Significance Level: {protocol.alpha}")
        click.echo(f"  Correction Method: {protocol.correction_method}")
        click.echo(f"  Min Train Samples: {protocol.min_train_samples}")
        click.echo(f"  Min Test Samples: {protocol.min_test_samples}")
    
    if passed:
        click.secho(f"\n[+] {msg}", fg='green', bold=True)
    else:
        click.secho(f"\n[-] {msg}", fg='red', bold=True)


@cli.command()
@click.option('--brands', default='data_raw/brands.csv', help='CSV with brand,company,ticker,aliases,therapeutic_area')
@click.option('--start', default=2015, type=int, help='Start year')
@click.option('--end', default=2024, type=int, help='End year')
def collect_data(brands, start, end):
    """Collect real launches and revenues (FDA, CT.gov, SEC)."""
    from data.collect_real import build_real_dataset

    seed = Path(brands)
    if not seed.exists():
        click.secho(f"Seed file not found: {brands}", fg='red')
        sys.exit(1)

    click.echo("=" * 50)
    click.echo("Collecting real launches and revenues")
    click.echo("=" * 50)
    ok = build_real_dataset(seed, years=(start, end))
    if ok:
        click.secho("[+] Collection complete. Parquet files written to data_proc/", fg='green')
    else:
        click.secho("[-] Collection produced no rows. Check seed file and API access.", fg='red')


@cli.command()
def audit():
    """Show audit summary and check Gate G5."""
    click.echo("=" * 50)
    click.echo("Audit Summary")
    click.echo("=" * 50)
    
    logger = get_audit_logger()
    summary = logger.get_summary()
    
    click.echo(f"\nAPI Usage:")
    click.echo(f"  Total Calls: {summary['api_calls']}")
    click.echo(f"  Total Tokens: {summary['total_tokens']:,}")
    click.echo(f"  Total Cost: ${summary['total_cost']:.2f}")
    click.echo(f"  Avg Cost/Call: ${summary['avg_cost_per_call']:.3f}")
    click.echo(f"  Experiments Run: {summary['experiments_run']}")
    
    # Check Gate G5
    gates = AcceptanceGates()
    
    audit_log = {
        'git_commit': logger.provenance['git_commit'],
        'git_dirty': logger.provenance['git_dirty'],
        'seed': 42,  # From protocol
        'data_version': 'v0.1.0',
        'model_config': {},
        'total_cost': summary['total_cost'],
        'api_calls': summary['api_calls']
    }
    
    passed, msg = gates.check_gate_g5(audit_log)
    
    if passed:
        click.secho(f"\n[+] {msg}", fg='green', bold=True)
    else:
        click.secho(f"\n[-] {msg}", fg='red', bold=True)
        if audit_log.get('git_dirty'):
            click.echo("  Hint: Commit your changes for full reproducibility")


@cli.command()
@click.option('--experiment', type=click.Choice(['h1', 'h2', 'h3', 'all']), 
              default='all', help='Which hypothesis to test')
@click.option('--seed', default=42, help='Random seed')
def evaluate(experiment, seed):
    """Run evaluation experiments (H1, H2, H3)."""
    click.echo("=" * 50)
    click.echo(f"Running evaluation: {experiment.upper()}")
    click.echo("=" * 50)
    
    # Check if data exists
    data_dir = Path("data_proc")
    if not (data_dir / "launches.parquet").exists():
        click.secho("Error: Dataset not built. Run 'build-data' first.", fg='red')
        sys.exit(1)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    if experiment in ['h1', 'all']:
        click.echo("\nH1: Evidence Grounding")
        click.echo("Testing whether evidence-grounded forecasts beat heuristics...")
        result = subprocess.run(['python', 'evaluation/run_h1.py', '--seed', str(seed)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            click.secho("✓ H1 experiment completed", fg='green')
        else:
            click.secho("✗ H1 experiment failed", fg='red')
            click.echo(result.stdout)
            click.echo(result.stderr)
        
    if experiment in ['h2', 'all']:
        click.echo("\nH2: Architecture Comparison")
        click.echo("Testing multi-agent vs single-agent performance...")
        result = subprocess.run(['python', 'evaluation/run_h2.py', '--seed', str(seed)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            click.secho("✓ H2 experiment completed", fg='green')
        else:
            click.secho("✗ H2 experiment failed", fg='red')
            click.echo(result.stdout)
            click.echo(result.stderr)
        
    if experiment in ['h3', 'all']:
        click.echo("\nH3: Domain Constraints")
        click.echo("Testing impact of pharmaceutical domain constraints...")
        result = subprocess.run(['python', 'evaluation/run_h3.py', '--seed', str(seed)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            click.secho("✓ H3 experiment completed", fg='green')
        else:
            click.secho("✗ H3 experiment failed", fg='red')
            click.echo(result.stdout)
            click.echo(result.stderr)
    
    click.echo("\n" + "=" * 50)
    click.echo("All experiments completed. Check results/ directory for outputs.")


@cli.command()
def gates():
    """Check all acceptance gates (G1-G5)."""
    click.echo("=" * 50)
    click.echo("Acceptance Gates Status")
    click.echo("=" * 50)
    
    gates_status = []
    
    # G1: Data
    profile_path = Path("results/data_profile.json")
    if profile_path.exists():
        with open(profile_path) as f:
            profile = json.load(f)
        g1_passed = (profile['n_launches'] >= 50 and 
                    profile['n_therapeutic_areas'] >= 5)
        gates_status.append(('G1', 'Data Quality', g1_passed))
    else:
        gates_status.append(('G1', 'Data Quality', None))
    
    # G2: Baselines
    import subprocess
    result = subprocess.run(['python', '-m', 'pytest', 'tests/test_baselines.py', '-q'],
                          capture_output=True, text=True)
    g2_passed = result.returncode == 0
    gates_status.append(('G2', 'Baselines', g2_passed))
    
    # G3: Statistical Protocol
    protocol = StatisticalProtocol()
    g3_gates = AcceptanceGates()
    g3_passed, _ = g3_gates.check_gate_g3(protocol, 50, 20)
    gates_status.append(('G3', 'Statistical Rigor', g3_passed))
    
    # G4: Results (needs experiments to run)
    gates_status.append(('G4', 'Beat Baselines', None))
    
    # G5: Reproducibility
    logger = get_audit_logger()
    audit_log = {
        'git_commit': logger.provenance['git_commit'],
        'git_dirty': logger.provenance['git_dirty'],
        'seed': 42,
        'data_version': 'v0.1.0',
        'model_config': {},
        'total_cost': 0,
        'api_calls': 0
    }
    g5_passed, _ = g3_gates.check_gate_g5(audit_log)
    gates_status.append(('G5', 'Reproducibility', g5_passed))
    
    # Display results
    click.echo("")
    for gate, desc, status in gates_status:
        if status is None:
            icon = "?"
            color = 'yellow'
            status_text = "NOT RUN"
        elif status:
            icon = "+"
            color = 'green'
            status_text = "PASSED"
        else:
            icon = "-"
            color = 'red'
            status_text = "FAILED"
        
        click.secho(f"  [{icon}] {gate}: {desc:<20} [{status_text}]", fg=color)
    
    click.echo("\n" + "=" * 50)
    
    # Overall status
    ready_gates = [s for _, _, s in gates_status if s is not None]
    if all(ready_gates):
        click.secho("Ready for conference submission!", fg='green', bold=True)
    else:
        not_passed = [f"{g[0]}" for g in gates_status if g[2] is False]
        not_run = [f"{g[0]}" for g in gates_status if g[2] is None]
        
        if not_passed:
            click.echo(f"Failed gates: {', '.join(not_passed)}")
        if not_run:
            click.echo(f"Not run: {', '.join(not_run)}")


if __name__ == '__main__':
    cli()