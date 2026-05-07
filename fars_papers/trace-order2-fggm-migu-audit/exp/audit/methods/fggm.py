# FGGM: Fisher-Guided Gradient Masking for continual learning.
# Computes diagonal FIM via model_engine.backward() only (no step), so model
# weights and optimizer state are NOT modified. Uses param.register_hook to
# capture pre-reduction gradients for Fisher accumulation.
# Applies Input-dimension Aggregation for weight matrices, thresholds at alpha
# quantile to build binary mask. alpha=0.7 => top 30% params updated.
# Uses param.register_hook for gradient masking (DeepSpeed ZeRO-2 compatible).

import os
import torch
import torch.nn as nn
import torch.distributed as dist


class FGGMHooks:

    def __init__(self, model_engine, train_dataloader, device, alpha=0.7,
                 task_idx=0, mask_save_path=None):
        self.model_engine = model_engine
        self.alpha = alpha
        self._bwd_handles = []
        self.masks = {}

        module = model_engine.module if hasattr(model_engine, 'module') else model_engine

        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[FGGM][rank={rank}] Computing Fisher information for task {task_idx}...", flush=True)

        fisher = self._compute_fisher(model_engine, module, train_dataloader, device)

        print(f"[FGGM][rank={rank}] Building binary masks (alpha={alpha})...", flush=True)
        self._build_masks(module, fisher)
        del fisher
        torch.cuda.empty_cache()

        if mask_save_path is not None and rank == 0:
            os.makedirs(mask_save_path, exist_ok=True)
            save_file = os.path.join(mask_save_path, f"task_{task_idx}.pt")
            torch.save({k: v.cpu() for k, v in self.masks.items()}, save_file)
            print(f"[FGGM] Masks saved to {save_file}", flush=True)

        self._register_hooks(module)
        print(f"[FGGM][rank={rank}] Gradient hooks registered. Training will proceed with masked gradients.", flush=True)

    def _compute_fisher(self, model_engine, module, train_dataloader, device):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        rank = dist.get_rank() if dist.is_initialized() else 0

        fisher_accum = {}
        param_name_map = {}
        for name, param in module.named_parameters():
            if param.requires_grad:
                fisher_accum[name] = torch.zeros(param.shape, dtype=torch.float32, device='cpu')
                param_name_map[id(param)] = name

        fisher_hooks = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                def make_hook(pid):
                    def hook(grad):
                        n = param_name_map[pid]
                        fisher_accum[n].add_((grad.detach().float() ** 2).cpu())
                        return grad
                    return hook
                h = param.register_hook(make_hook(id(param)))
                fisher_hooks.append(h)

        model_engine.train()
        total_batches = 0

        for batch in train_dataloader:
            for key in ["sources", "gts"]:
                if key in batch:
                    del batch[key]
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model_engine(**batch, use_cache=False)
            loss = outputs.loss
            model_engine.backward(loss)
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            total_batches += 1
            del outputs, loss, batch
            if total_batches % 10 == 0:
                torch.cuda.empty_cache()
                if rank == 0:
                    print(f"[FGGM] Fisher batch {total_batches}/{len(train_dataloader)}", flush=True)

        for h in fisher_hooks:
            h.remove()

        for name in fisher_accum:
            fisher_accum[name] /= max(total_batches, 1)

        if dist.is_initialized():
            for name in fisher_accum:
                fisher_accum[name] = fisher_accum[name].to(device)
                dist.all_reduce(fisher_accum[name], op=dist.ReduceOp.AVG)

        gc.collect()
        torch.cuda.empty_cache()

        print(f"[FGGM][rank={rank}] Fisher computed over {total_batches} batches (no model/optim modification).", flush=True)
        return fisher_accum

    def _build_masks(self, module, fisher):
        aggregated_scores = []

        param_agg = {}
        for name, param in module.named_parameters():
            if name not in fisher:
                continue
            f = fisher[name]
            if f.dim() == 2:
                agg = f.sum(dim=1)
            else:
                agg = f
            param_agg[name] = agg
            aggregated_scores.append(agg)

        all_scores = torch.cat(aggregated_scores)
        quantile_val = torch.quantile(all_scores, self.alpha)

        rank = dist.get_rank() if dist.is_initialized() else 0
        total_elements = all_scores.numel()
        updated_count = (all_scores > quantile_val).sum().item()
        frozen_count = total_elements - updated_count
        print(f"[FGGM][rank={rank}] Global quantile({self.alpha:.2f}) = {quantile_val:.6e}, "
              f"updated {updated_count}/{total_elements} ({updated_count/total_elements*100:.1f}%), "
              f"frozen {frozen_count}/{total_elements} ({frozen_count/total_elements*100:.1f}%)", flush=True)

        for name, param in module.named_parameters():
            if name not in param_agg:
                continue
            agg = param_agg[name]
            binary = (agg > quantile_val).to(dtype=param.dtype, device=param.device)
            if param.dim() == 2:
                binary = binary.unsqueeze(1)
            self.masks[name] = binary

    def _register_hooks(self, module):
        for name, param in module.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                h = param.register_hook(
                    lambda grad, m=mask: grad * m
                )
                self._bwd_handles.append(h)

    def step_done(self):
        pass

    def remove(self):
        for h in self._bwd_handles:
            h.remove()
        self._bwd_handles.clear()
        self.masks.clear()


def get_hooks(config):
    alpha = config.get("fggm_alpha", 0.7)
    mask_save_dir = config.get("mask_save_dir", None)

    def _make_hooks(model_engine, task_idx, **kwargs):
        if task_idx == 0:
            return None
        train_dataloader = kwargs.get("train_dataloader")
        device = kwargs.get("device")
        if train_dataloader is None or device is None:
            raise ValueError("FGGM requires train_dataloader and device kwargs")
        return FGGMHooks(
            model_engine, train_dataloader, device,
            alpha=alpha, task_idx=task_idx,
            mask_save_path=mask_save_dir,
        )

    return _make_hooks
