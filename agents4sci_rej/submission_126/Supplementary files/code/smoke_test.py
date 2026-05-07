
from run_experiments import run_comprehensive_experiments

if __name__ == '__main__':
    print("Running smoke test...")
    run_comprehensive_experiments(smoke_test=True)
    print("Smoke test finished.")
