"""
Provenance tracking for reproducibility.
Following Linus principle: Simple, direct, logs what matters.
"""
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()[:8]  # Short hash
    except:
        return "unknown"

def get_git_branch() -> str:
    """Get current git branch."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"

def log_experiment_provenance(
    experiment_name: str,
    seed: int,
    parameters: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Log experiment provenance for reproducibility.
    
    Args:
        experiment_name: Name of the experiment
        seed: Random seed used
        parameters: Experiment parameters
        output_dir: Directory to save provenance log
    
    Returns:
        Dict with provenance information
    """
    provenance = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
        "python_version": os.sys.version,
        "platform": os.sys.platform,
        "parameters": parameters
    }
    
    # Log to console
    print(f"[PROVENANCE] Experiment: {experiment_name}")
    print(f"[PROVENANCE] Seed: {seed}")
    print(f"[PROVENANCE] Git: {provenance['git_branch']}@{provenance['git_commit']}")
    print(f"[PROVENANCE] Time: {provenance['timestamp']}")
    
    # Save to file if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"provenance_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(provenance, f, indent=2, default=str)
        
        print(f"[PROVENANCE] Saved to: {filepath}")
    
    return provenance

def set_reproducible_seed(seed: int = 42) -> None:
    """
    Set seed for all random number generators.
    
    Args:
        seed: Random seed to use
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Log the seed setting
    print(f"[SEED] Random seed set to: {seed}")
    
    # Try to set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[SEED] PyTorch seed set to: {seed}")
    except ImportError:
        pass