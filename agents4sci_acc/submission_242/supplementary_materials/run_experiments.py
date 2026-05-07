#!/usr/bin/env python3
"""
Run marketplace experiments with different configurations.
"""

import logging
from pathlib import Path
from datetime import datetime
from src.marketplace.true_gpt_marketplace import TrueGPTMarketplace
from src.analysis.market_analysis import MarketplaceAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"results/experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_experiment(config: dict) -> dict:
    """Run a single experiment with the given configuration"""
    try:
        # Initialize marketplace
        marketplace = TrueGPTMarketplace(
            num_freelancers=config.get("num_freelancers", 5),
            num_clients=config.get("num_clients", 3),
            rounds=config.get("rounds", 10)
        )
        
        # Run simulation
        simulation_results = marketplace.run_simulation()
        
        # Run analysis
        output_dir = Path(config.get("output_dir", "results"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the simulation results file
        simulation_file = next(Path("results/simuleval").glob("true_gpt_simulation_*.json"))
        analyzer = MarketplaceAnalysis(str(simulation_file))
        analysis_results = analyzer.run_full_analysis(output_dir=str(output_dir))
        
        return {
            "success": True,
            "simulation": simulation_results,
            "analysis": analysis_results,
            "config": config
        }
        
    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "config": config
        }

def main():
    """Run a series of experiments with different configurations"""
    logger.info("Starting marketplace experiments")
    
    # Create results directory
    results_dir = Path("results/experiments")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define experiment configurations
    experiments = [
        {
            "name": "baseline",
            "rounds": 10,
            "num_freelancers": 5,
            "num_clients": 3,
            "output_dir": str(results_dir / "baseline")
        },
        {
            "name": "large_market",
            "rounds": 15,
            "num_freelancers": 10,
            "num_clients": 5,
            "output_dir": str(results_dir / "large_market")
        },
        {
            "name": "extended_time",
            "rounds": 20,
            "num_freelancers": 5,
            "num_clients": 3,
            "output_dir": str(results_dir / "extended_time")
        }
    ]
    
    # Run experiments
    results = []
    for config in experiments:
        logger.info(f"Running experiment: {config['name']}")
        result = run_experiment(config)
        results.append(result)
        
        if result["success"]:
            logger.info(f"Experiment {config['name']} completed successfully")
        else:
            logger.error(f"Experiment {config['name']} failed: {result['error']}")
    
    # Log summary
    successful = sum(1 for r in results if r["success"])
    logger.info(f"\nExperiments completed: {successful}/{len(experiments)} successful")
    
    return results

if __name__ == "__main__":
    main()