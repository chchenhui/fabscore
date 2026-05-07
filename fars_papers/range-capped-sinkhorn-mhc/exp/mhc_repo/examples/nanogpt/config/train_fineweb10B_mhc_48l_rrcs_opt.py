# Optimized RRCS config with reduced r_cap for actual gradient flow.
# mhc_r_cap is overridden at launch time via CLI: mhc_r_cap=<value>

out_dir = "out-fineweb10B-mhc-48l-rrcs-opt"
wandb_run_name = "rrcs-opt-48l"
wandb_project = "mhc-nanogpt-48"

dataset = "fineweb10B"

# model
block_size = 1024
n_layer = 48
n_head = 6
n_embd = 150
dropout = 0.0
bias = False

batch_size = 8
gradient_accumulation_steps = 4
max_iters = 5000
eval_interval = 500
log_interval = 10
eval_iters = 100

# optimizer
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 6e-5

# dtype
dtype = "bfloat16"

# hyper-connections: mHC enabled (4 streams) with RRCS
hc_num_streams = 4
hc_num_fracs = 1
hc_disable = False
mhc = True
sinkhorn_iters = 10
sinkhorn_tau = 0.05
mhc_h_res_proj = "sinkhorn"
ns_steps = 5
ns_eps = 1e-7
ns_coeffs = (3.0, -3.2, 1.2)
mhc_rrcs = True
mhc_r_cap = 5.0
