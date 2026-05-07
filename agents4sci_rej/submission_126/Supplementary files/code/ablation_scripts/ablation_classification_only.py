import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_experiments import run_comprehensive_experiments

if __name__ == '__main__':
    run_comprehensive_experiments(
        smoke_test=False,
        model_size='small',
        backbone_type='vit',
        ablation_tasks=['classification'],
        use_cross_attention=True,
        kp_head_type='hrnet',
        use_learnable_loss=True,
        fixed_loss_weights=None,
        output_dir='ablation_results/Classification_Only'
    )