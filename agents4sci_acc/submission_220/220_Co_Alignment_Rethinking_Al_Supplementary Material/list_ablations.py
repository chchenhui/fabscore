import yaml

with open('bica/configs/maptalk_ablation.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('🧪 Complete Ablation Study - All Variants:')
print('=' * 50)

for i, (variant_name, variant_config) in enumerate(config['ablation_variants'].items(), 1):
    exp_name = variant_config.get('experiment_name', variant_name)
    print(f'{i:2d}. {variant_name:<20} -> {exp_name}')

print(f'\nTotal: {len(config["ablation_variants"])} variants')
print('\n🔧 Variant Type Analysis:')
print('- Architecture ablations: small_code, large_code, no_gru, small_hidden, large_hidden')
print('- Hyperparameter ablations: high_temp, low_temp, tight_budgets, loose_budgets, low_ib, high_ib')
print('- Co-alignment ablations: no_rep_gap, high_rep_gap, no_instructor_cost, high_instructor_cost')
