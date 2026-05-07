# Setting A: HC Unconstrained (no doubly-stochastic projection), 48-layer, hc_num_streams=4
# Based on train_fineweb10B_hc_48l.py - calibration run for gradient spike baseline

out_dir = "out-setting-a-hc-unconstrained"
wandb_run_name = "setting-a-hc-unconstrained"
wandb_project = "orthostochastic-mhc"

dataset = "fineweb10B"

block_size = 1024
n_layer = 48
n_head = 6
n_embd = 150
dropout = 0.0
bias = False

batch_size = 8
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

hc_num_streams = 4
hc_num_fracs = 1
hc_disable = False

max_train_shards = 10
