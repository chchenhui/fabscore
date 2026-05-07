#!/usr/bin/env python3
"""
Experiment Execution Entry Point Script

This script provides a command line interface for running experimental workflows
using the experiment platform. It supports experiment design, execution, and analysis.

Results are organized using the groups/runs structure within the specified workspace:
- groups/{group_id}/runs/{timestamp}/ for each experimental group
- No separate output directory is needed - everything is organized within the workspace
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onesim.models import get_model_manager, get_model
from loguru import logger

# Import experiment platform components
from researcher.experiment_platform.experiment_project import ExperimentProject, ProjectConfig


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Execute experimental workflows using the OneSim experiment platform."
    )
    
    # Required arguments
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to experiment_config.json file.",
        required=True
    )
    
    parser.add_argument(
        "--intervention_specs",
        type=str,
        help="Path to intervention_specifications.json file.",
        required=True
    )
    
    parser.add_argument(
        "--workspace_path",
        type=str,
        help="Path to experiment workspace directory (e.g., src/researcher/experiments/dynamic_culture_dissemination_impact). Results will be organized in groups/runs structure within this workspace.",
        required=True
    )
    
    parser.add_argument(
        "--environment",
        type=str,
        help="Path to base environment directory (e.g., src/envs/dynamic_culture_dissemination). If not specified, will be inferred from workspace.",
        required=False
    )
    
    # Optional arguments
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model configuration name (config_name from JSON) to use."
    )
    
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/model_config.json",
        help="Path to the model configuration JSON file (relative to project root)."
    )
    
    parser.add_argument(
        "--skip_simulation",
        action="store_true",
        help="Skip simulation execution, only prepare experiment files.",
        default=False
    )
    
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate the experiment setup without executing.",
        default=False
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform a dry run without creating files or executing simulations.",
        default=False
    )
    
    return parser.parse_args()


def get_project_root() -> Path:
    """Get the project root directory (assuming this script is in src/researcher/)."""
    return Path(__file__).resolve().parent.parent.parent


def validate_inputs(args) -> Dict[str, Any]:
    """
    Validate input files and paths.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dict containing validation results and resolved paths
    """
    project_root = get_project_root()
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'paths': {}
    }
    
    # Validate experiment config
    experiment_config_path = Path(args.experiment_config)
    if not experiment_config_path.is_absolute():
        experiment_config_path = project_root / args.experiment_config
    
    if not experiment_config_path.exists():
        results['errors'].append(f"Experiment config file not found: {experiment_config_path}")
        results['valid'] = False
    else:
        results['paths']['experiment_config'] = str(experiment_config_path)
    
    # Validate intervention specs
    intervention_specs_path = Path(args.intervention_specs)
    if not intervention_specs_path.is_absolute():
        intervention_specs_path = project_root / args.intervention_specs
    
    if not intervention_specs_path.exists():
        results['errors'].append(f"Intervention specs file not found: {intervention_specs_path}")
        results['valid'] = False
    else:
        results['paths']['intervention_specs'] = str(intervention_specs_path)
    
    # Validate workspace path
    workspace_path = Path(args.workspace_path)
    if not workspace_path.is_absolute():
        workspace_path = project_root / args.workspace_path
    
    if not workspace_path.exists():
        results['errors'].append(f"Workspace directory not found: {workspace_path}")
        results['valid'] = False
    elif not workspace_path.is_dir():
        results['errors'].append(f"Workspace path is not a directory: {workspace_path}")
        results['valid'] = False
    else:
        results['paths']['workspace'] = str(workspace_path)
        
    
    # Validate environment path (if provided)
    if args.environment:
        env_path = Path(args.environment)
        if not env_path.is_absolute():
            env_path = project_root / args.environment
        
        if not env_path.exists():
            results['errors'].append(f"Environment directory not found: {env_path}")
            results['valid'] = False
        elif not env_path.is_dir():
            results['errors'].append(f"Environment path is not a directory: {env_path}")
            results['valid'] = False
        else:
            results['paths']['environment'] = str(env_path)
    else:
        # Infer environment from experiment_config.json if not provided
        if results['paths'].get('experiment_config'):
            try:
                with open(results['paths']['experiment_config'], 'r', encoding='utf-8') as f:
                    exp_config = json.load(f)
                env_source = exp_config.get('base_scenario', {}).get('environment')

                if env_source:
                    inferred_env_path = project_root / "src" / "envs" / env_source
                    if inferred_env_path.exists() and inferred_env_path.is_dir():
                        results['paths']['environment'] = str(inferred_env_path)
                        results['warnings'].append(f"Using inferred environment path from experiment_config.json: {inferred_env_path}")
                    else:
                        results['errors'].append(f"Inferred environment path '{inferred_env_path}' from experiment_config.json not found or not a directory.")
                        results['valid'] = False
                else:
                    results['errors'].append("'environment' key not found in 'base_scenario' of experiment_config.json.")
                    results['valid'] = False
            except (json.JSONDecodeError, KeyError) as e:
                results['errors'].append(f"Error parsing experiment_config.json to infer environment: {e}")
                results['valid'] = False
        else:
            results['warnings'].append("No environment path provided and cannot infer from experiment config.")
    
    # Validate model config if provided
    if args.model_config:
        model_config_path = Path(args.model_config)
        if not model_config_path.is_absolute():
            model_config_path = project_root / args.model_config
        
        if not model_config_path.exists():
            results['errors'].append(f"Model config file not found: {model_config_path}")
            results['valid'] = False
        else:
            results['paths']['model_config'] = str(model_config_path)
    
    return results


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate experiment configuration.
    
    Args:
        config_path: Path to experiment config file
        
    Returns:
        Loaded experiment configuration
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Basic validation
    required_sections = ['experiment_info', 'experimental_groups']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in experiment config")
    
    if 'name' not in config['experiment_info']:
        raise ValueError("Missing 'name' in experiment_info")
    
    return config


def run_experiment_workflow(args, validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the complete experimental workflow.
    
    Args:
        args: Command line arguments
        validation_results: Results from input validation
        
    Returns:
        Workflow execution results
    """
    paths = validation_results['paths']
    
    # Load experiment configuration
    logger.info("Loading experiment configuration...")
    experiment_config = load_experiment_config(paths['experiment_config'])
    experiment_name = experiment_config['experiment_info']['name']
    
    # Output directly to workspace (no separate output_dir)
    # The experiment platform will handle creating groups/runs structure
    output_path = Path(paths['workspace'])
    
    if args.dry_run:
        logger.info(f"[DRY RUN] Would use workspace directory: {output_path}")
        logger.info(f"[DRY RUN] Groups/runs structure would be created within workspace")
        return {
            'status': 'dry_run_completed',
            'experiment_name': experiment_name,
            'workspace_path': str(output_path)
        }
    
    # Create project configuration
    # Use standard OneSim configuration as base, not scenario config
    project_root = get_project_root()
    base_config_path = project_root / "config" / "config.json"
    
    # Fallback to workspace config if standard config doesn't exist
    if not base_config_path.exists():
        base_config_path = Path(paths['workspace']) / "config.json"
        if not base_config_path.exists():
            raise FileNotFoundError(f"Could not find base configuration file. Tried: {project_root / 'config' / 'config.json'} and {Path(paths['workspace']) / 'config.json'}")
    
    # Use workspace as base_environment and environment path for actual environment
    base_environment = paths['workspace']
    if 'environment' in paths:
        # If environment is specified, use it for environment-specific operations
        env_path = paths['environment']
    else:
        # If no environment specified, use workspace (for backward compatibility)
        env_path = paths['workspace']
    
    project_config = ProjectConfig(
        project_name=experiment_name,
        project_description=experiment_config['experiment_info'].get('description', ''),
        base_environment=base_environment,
        base_config_path=str(base_config_path),
        intervention_specs_path=paths['intervention_specs'],
        output_directory=str(output_path),
        model_name=args.model_name,
        model_config_path=paths.get('model_config')
    )
    
    # Initialize experiment project
    logger.info(f"Initializing experiment project: {experiment_name}")
    logger.info(f"Workspace directory: {output_path}")
    logger.info(f"Results will be organized in groups/runs structure within workspace")
    
    experiment_project = ExperimentProject(project_config)
    
    # Validation mode
    if args.validate_only:
        logger.info("Running validation checks...")
        validation_results = experiment_project.validate_project()
        
        if validation_results['valid']:
            logger.info("✅ Experiment validation passed!")
        else:
            logger.error("❌ Experiment validation failed!")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
        
        return {
            'status': 'validation_completed',
            'validation_results': validation_results
        }
    
    # Run the complete workflow
    logger.info("Starting experiment workflow...")
    if args.skip_simulation:
        # Run workflow without simulation
        logger.info("Skipping simulation execution...")
        
        # Step 1-6: Everything except simulation
        workflow_results = {
            'project_name': experiment_name,
            'status': 'running',
            'steps_completed': []
        }
        
        try:
            # Load base profiles
            logger.info("Step 1: Loading base profiles...")
            experiment_project.load_base_profiles(env_path)
            workflow_results['steps_completed'].append('load_profiles')
            
            # Parse intervention assignments
            logger.info("Step 2: Parsing intervention assignments...")
            intervention_configs = experiment_project.parse_intervention_assignments(experiment_config)
            workflow_results['steps_completed'].append('parse_interventions')
            
            # Design experiment
            logger.info("Step 3: Designing experiment...")
            experiment_groups = experiment_project.design_experiment(
                experiment_config.get('experimental_groups', {})
            )
            workflow_results['steps_completed'].append('design_experiment')
            
            # Apply interventions
            logger.info("Step 4: Applying interventions and setting up profiles...")
            intervention_results = experiment_project.apply_interventions(
                experiment_groups, intervention_configs, env_path
            )
            workflow_results['steps_completed'].append('apply_interventions')
            
            # Generate configurations
            logger.info("Step 5: Generating configurations...")
            generated_configs = experiment_project.generate_configurations(experiment_groups)
            workflow_results['steps_completed'].append('generate_configurations')
            
            # Save summary
            logger.info("Step 6: Saving project summary...")
            experiment_project._save_project_summary(workflow_results, experiment_groups, intervention_results)
            workflow_results['steps_completed'].append('save_summary')
            
            workflow_results['status'] = 'completed_without_simulation'
            workflow_results['workspace_directory'] = str(output_path)
            workflow_results['configurations_generated'] = len(generated_configs)
            
        except Exception as e:
            workflow_results['status'] = 'failed'
            workflow_results['error'] = str(e)
            logger.error(f"Workflow failed: {e}")
            raise
            
    else:
        # Run full workflow including simulation
        workflow_results = experiment_project.run_full_workflow_from_config(
            environment_path=env_path,
            experiment_config_path=paths['experiment_config']
        )
    
    return workflow_results


def display_results(results: Dict[str, Any]):
    """
    Display workflow results in a user-friendly format.
    
    Args:
        results: Workflow execution results
    """
    status = results.get('status', 'unknown')
    
    if status == 'dry_run_completed':
        logger.info("🏃 Dry run completed successfully!")
        logger.info(f"📁 Would use workspace: {results['workspace_path']}")
        logger.info(f"📁 Groups/runs structure would be created within workspace")
        logger.info(f"🧪 Experiment: {results['experiment_name']}")
        
    elif status == 'validation_completed':
        validation_results = results['validation_results']
        if validation_results['valid']:
            logger.info("✅ Validation completed successfully!")
            logger.info(f"🔍 Components checked: {len(validation_results['components_checked'])}")
            if validation_results.get('warnings'):
                logger.info(f"⚠️  Warnings: {len(validation_results['warnings'])}")
        else:
            logger.error("❌ Validation failed!")
            logger.error(f"🔍 Errors found: {len(validation_results['errors'])}")
            
    elif status in ['completed', 'completed_without_simulation']:
        logger.info("✅ Experiment workflow completed successfully!")
        workspace_dir = results.get('workspace_directory') or results.get('output_directory', 'Unknown')
        logger.info(f"📁 Workspace directory: {workspace_dir}")
        logger.info(f"📁 Results organized in groups/runs structure within workspace")
        logger.info(f"📊 Groups processed: {results.get('experiment_groups_count', 0)}")
        logger.info(f"🔧 Interventions applied: {results.get('interventions_applied', 0)}")
        logger.info(f"⚙️  Configurations generated: {results.get('configurations_generated', 0)}")
        
        if status == 'completed':
            # Show simulation results
            sim_results = results.get('simulation_results', {})
            if sim_results:
                total_sims = sim_results.get('total_simulations', 0)
                successful_sims = sim_results.get('successful_simulations', 0)
                logger.info(f"🎯 Simulations executed: {successful_sims}/{total_sims} successful")
                
                # Show individual simulation results
                sim_details = sim_results.get('simulation_details', {})
                if sim_details:
                    logger.info("\n🎮 Simulation results:")
                    for group_id, detail in sim_details.items():
                        status_emoji = "✅" if detail['status'] == 'completed' else "❌"
                        logger.info(f"  {status_emoji} {group_id}: {detail['status']}")
                        if detail.get('execution_time'):
                            logger.info(f"    ⏱️  Time: {detail['execution_time']:.2f}s")
                        if detail.get('error'):
                            logger.info(f"    ❌ Error: {detail['error']}")
        else:
            logger.info("⏸️  Simulations skipped (configurations ready for manual execution)")
            
    elif status == 'failed':
        logger.error("❌ Experiment workflow failed!")
        error = results.get('error', 'Unknown error')
        logger.error(f"💥 Error: {error}")
        
    else:
        logger.warning(f"⚠️  Unknown status: {status}")


def main():
    """
    Main entry point for the experiment execution workflow.
    """
    args = parse_args()
    
    try:
        # Validate inputs
        logger.info("Validating experiment inputs...")
        validation_results = validate_inputs(args)
        
        if not validation_results['valid']:
            logger.error("❌ Input validation failed!")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
            return 1
        
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                logger.warning(f"⚠️  {warning}")
        
        logger.info("✅ Input validation passed!")
        
        # Setup model manager if needed
        if not args.dry_run and not args.validate_only:
            if 'model_config' in validation_results['paths']:
                model_config_path = validation_results['paths']['model_config']
                model_manager = get_model_manager()
                model_manager.load_model_configs(str(model_config_path))
                logger.info(f"Loaded model configurations from: {model_config_path}")
        
        # Execute workflow
        logger.info(f"{'='*80}")
        logger.info(f"EXPERIMENT EXECUTION WORKFLOW")
        logger.info(f"{'='*80}")
        
        results = run_experiment_workflow(args, validation_results)
        
        # Display results
        logger.info(f"\n{'='*80}")
        logger.info(f"WORKFLOW RESULTS")
        logger.info(f"{'='*80}")
        
        display_results(results)
        
        logger.info(f"{'='*80}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Experiment execution failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())




