@echo off
REM Run experiments with multiple seeds in parallel, starting from the same initial weights

SET SEEDS=10 20 30 40 50

REM Create a directory for the results
md ..\Results\multiseed_finetune

REM Run the experiments in parallel
for %%s in (%SEEDS%) do (
    echo "Starting experiment with seed %%s"
    start /b python run_experiments.py --seed %%s --output_file ..\Results\multiseed_finetune\results_%%s.json --num-epochs 50 --load-weights-from ..\Results\best_model.pth
)

echo "All experiments started."