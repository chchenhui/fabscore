# Setting B: mHC-Orthostochastic, 6-layer, hc_num_streams=8
# Optimized: ns_steps=20 (better convergence at n=8), identity_mix=True (alpha=0.1),
# eval_interval=250 (finer checkpoint selection)

out_dir = "out-setting-b-orthostochastic"
wandb_run_name = "setting-b-orthostochastic"
wandb_project = "orthostochastic-mhc"

dataset = "fineweb10B"

block_size = 1024
n_layer = 6
n_head = 6
n_embd = 288
dropout = 0.0
bias = False

batch_size = 32
gradient_accumulation_steps = 4
max_iters = 5000
eval_interval = 250
log_interval = 10
eval_iters = 100

learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 200
lr_decay_iters = 5000
min_lr = 6e-5

dtype = "bfloat16"

hc_num_streams = 8
hc_num_fracs = 1
hc_disable = False
mhc = True
sinkhorn_iters = 10
sinkhorn_tau = 0.05
mhc_h_res_proj = "orthostochastic"
ns_steps = 20
ns_eps = 1e-7
ns_coeffs = (3.0, -3.2, 1.2)

mhc_residual_identity_mix = True
mhc_residual_alpha = 0.1

max_train_shards = 10
