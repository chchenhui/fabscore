import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_experiments import run_comprehensive_experiments

if __name__ == '__main__':
    run_comprehensive_experiments(
        smoke_test=False,
        model_size='small',
        backbone_type='vit',
        ablation_tasks=['keypoint', 'classification', 'vhs'],
        use_cross_attention=True,
        kp_head_type='hrnet',
        use_learnable_loss=False,
        fixed_loss_weights=[2.0, 1.0, 1.0],
        output_dir='ablation_results/Fixed_Loss_Weights_KP_Heavy'
    )