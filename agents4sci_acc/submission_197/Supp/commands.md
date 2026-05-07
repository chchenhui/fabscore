# Experiment Commands for Raw Data Collection

## 1. Baseline (No Boids Rules)
```bash
python run_real_experiment.py --meta_prompt_id data_science_suite --mode single --num_agents 10 --num_rounds 10 --no_self_reflection
```

## 2. Boids with All Three Rules
```bash
python run_real_experiment.py --meta_prompt_id data_science_suite --mode single --num_agents 10 --num_rounds 10 --boids_enabled --boids_k 2 --boids_sep 0.45
```

## 3. Boids with Separation Rule Only
```bash
python run_real_experiment.py --meta_prompt_id data_science_suite --mode single --num_agents 10 --num_rounds 10 --boids_enabled --boids_separation --boids_k 2 --boids_sep 0.45 --no_self_reflection
```

## 4. Boids with Alignment Rule Only
```bash
python run_real_experiment.py --meta_prompt_id data_science_suite --mode single --num_agents 10 --num_rounds 10 --boids_enabled --boids_alignment --boids_k 2 --boids_sep 0.45 --no_self_reflection
```

## 5. Boids with Cohesion Rule Only
```bash
python run_real_experiment.py --meta_prompt_id data_science_suite --mode single --num_agents 10 --num_rounds 10 --boids_enabled --boids_cohesion --boids_k 2 --boids_sep 0.45
```

## 6. Boids with All Three Rules and Evolution
```bash
python run_real_experiment.py --meta_prompt_id data_science_suite --mode single --num_agents 10 --num_rounds 10 --boids_enabled --boids_k 2 --boids_sep 0.45 --evolution_enabled --evolution_frequency 2 --evolution_selection_rate 0.2
```