import os
import sys
import numpy as np

# Add src/ to Python path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils.logging_utils import run_with_emissions

def workload():
    """Small workload for energy measurement sanity check"""
    a = np.random.rand(500, 500)
    b = np.random.rand(500, 500)
    np.dot(a, b)

if __name__ == "__main__":
    os.makedirs("outputs/sanity", exist_ok=True)

    print("OPENBLAS_NUM_THREADS:", os.getenv("OPENBLAS_NUM_THREADS"))
    print("MKL_NUM_THREADS:", os.getenv("MKL_NUM_THREADS"))
    print("OMP_NUM_THREADS:", os.getenv("OMP_NUM_THREADS"))

    run_with_emissions("outputs/sanity", "sanity_check", workload)

    print("✅ Sanity check complete. Logs written to outputs/sanity/")

