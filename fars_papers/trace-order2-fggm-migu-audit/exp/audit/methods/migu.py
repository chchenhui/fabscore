# MIGU: Magnitude-based Gradient Updating for continual learning.
# Forward hooks cache output activation magnitudes per linear layer.
# Backward param hooks mask gradients to keep only top (1-T) fraction by magnitude.
# T=0.7 means 70% of output dimensions are masked (frozen), 30% are updated.
# Uses param.register_hook for DeepSpeed ZeRO-2 compatibility (module.weight.grad
# is None under ZeRO-2, so post-backward masking via module.weight.grad doesn't work).
# Reference: audit/external/MIGU/src/accelerate_local.py and uie_trainer_ft.py

import torch
import torch.nn as nn
import torch.distributed as dist


class MIGUHooks:

    def __init__(self, model_engine, threshold=0.7):
        self.model_engine = model_engine
        self.threshold = threshold
        self.temporal_activation_sum = {}
        self._fwd_handles = []
        self._bwd_handles = []

        module = model_engine.module if hasattr(model_engine, 'module') else model_engine
        for name, mod in module.named_modules():
            if isinstance(mod, nn.Linear):
                fh = mod.register_forward_hook(
                    lambda m, inp, out, n=name: self._forward_hook(n, m, inp, out)
                )
                self._fwd_handles.append(fh)

                bh = mod.weight.register_hook(
                    lambda grad, n=name, m=mod: self._grad_hook(n, m, grad)
                )
                self._bwd_handles.append(bh)

    def _forward_hook(self, name, module, input, output):
        hidden_dim = output.shape[-1]
        activation = torch.sum(output.reshape(-1, hidden_dim).abs(), dim=0)
        if name not in self.temporal_activation_sum:
            self.temporal_activation_sum[name] = activation
        else:
            self.temporal_activation_sum[name] += activation

    def _grad_hook(self, name, module, grad):
        if name not in self.temporal_activation_sum:
            return grad

        activation = self.temporal_activation_sum[name].clone().float()
        if dist.is_initialized():
            dist.all_reduce(activation, op=dist.ReduceOp.AVG)
        threshold_val = torch.quantile(activation, self.threshold)
        importance_mask = (activation >= threshold_val).to(dtype=grad.dtype, device=grad.device)
        masked_grad = grad * importance_mask.unsqueeze(dim=-1)
        return masked_grad

    def step_done(self):
        self.temporal_activation_sum = {}

    def remove(self):
        for h in self._fwd_handles:
            h.remove()
        for h in self._bwd_handles:
            h.remove()
        self._fwd_handles.clear()
        self._bwd_handles.clear()
        self.temporal_activation_sum = {}


def get_hooks(config):
    threshold = config.get("migu_threshold", 0.7)
    def _make_hooks(model_engine, task_idx, **kwargs):
        if task_idx == 0:
            return None
        return MIGUHooks(model_engine, threshold=threshold)
    return _make_hooks
