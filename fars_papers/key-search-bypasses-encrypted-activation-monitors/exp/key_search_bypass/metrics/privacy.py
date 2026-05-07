# Privacy metrics: KNN ASR@10 measurement for activation space recovery attacks.
# For each token position, check if the ground-truth token is among the top-10
# nearest neighbors of the encrypted embedding z_t in the LLM's embedding table.

import torch
from key_search_bypass.encryptor.model import KeyConditionedEncryptor
from key_search_bypass.encryptor.losses import _forward_from_embeds


@torch.no_grad()
def compute_knn_asr(model, encryptor, input_ids, attention_mask, key_dim=128, k=10, batch_size=4):
    """Compute KNN ASR@k: fraction of token positions where the ground-truth token
    appears among the k nearest neighbors of z_t in the embedding table.

    Args:
        model: frozen LLM with model.model.embed_tokens.weight as vocab embeddings
        encryptor: trained KeyConditionedEncryptor
        input_ids: (N, T) token ids
        attention_mask: (N, T) attention mask
        key_dim: key dimension
        k: number of nearest neighbors
        batch_size: eval batch size
    Returns:
        asr: float, fraction of token positions where ground-truth is in top-k
    """
    encryptor.eval()
    embed_table = model.model.embed_tokens.weight.float()
    embed_table_norm = embed_table / embed_table.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    total_hits = 0
    total_tokens = 0
    n = input_ids.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = input_ids[start:end].cuda()
        mask = attention_mask[start:end].cuda()

        clean_embeds = model.model.embed_tokens(ids)
        key = KeyConditionedEncryptor.sample_key(ids.shape[0], key_dim, device=ids.device, dtype=clean_embeds.dtype)
        z = encryptor(clean_embeds, key)

        z_flat = z.view(-1, z.shape[-1]).float()
        mask_flat = mask.view(-1).bool()
        ids_flat = ids.view(-1)

        z_active = z_flat[mask_flat]
        ids_active = ids_flat[mask_flat]

        dists = torch.cdist(z_active, embed_table, p=2)
        _, topk_indices = dists.topk(k, dim=-1, largest=False)

        hits = (topk_indices == ids_active.unsqueeze(1)).any(dim=1)
        total_hits += hits.sum().item()
        total_tokens += hits.shape[0]

    return total_hits / max(total_tokens, 1)


@torch.no_grad()
def compute_key_diversity(model, encryptor, input_ids, attention_mask, key_dim=128, n_prompts=200, n_keys=32, batch_size=4):
    """For each prompt, sample n_keys keys and compute pairwise L2 distances
    between encrypted embeddings. Report mean and std of all pairwise distances.

    Args:
        model: frozen LLM
        encryptor: trained KeyConditionedEncryptor
        input_ids: (N, T) token ids
        attention_mask: (N, T) attention mask
        key_dim: key dimension
        n_prompts: number of prompts to use
        n_keys: number of keys per prompt
        batch_size: not used for pairwise computation, kept for API consistency
    Returns:
        mean_dist, std_dist: floats
    """
    encryptor.eval()
    n = min(n_prompts, input_ids.shape[0])
    all_dists = []

    for i in range(n):
        ids = input_ids[i:i+1].cuda()
        mask = attention_mask[i:i+1].cuda()

        with torch.no_grad():
            clean_embeds = model.model.embed_tokens(ids)

        embeddings_list = []
        for _ in range(n_keys):
            key = KeyConditionedEncryptor.sample_key(1, key_dim, device=ids.device, dtype=clean_embeds.dtype)
            z = encryptor(clean_embeds, key)
            seq_len = mask.sum().item()
            z_mean = z[0, :int(seq_len)].mean(dim=0)
            embeddings_list.append(z_mean)

        z_stack = torch.stack(embeddings_list, dim=0).float()
        pairwise = torch.cdist(z_stack.unsqueeze(0), z_stack.unsqueeze(0), p=2).squeeze(0)
        triu_mask = torch.triu(torch.ones(n_keys, n_keys, device=pairwise.device), diagonal=1).bool()
        dists = pairwise[triu_mask]
        all_dists.append(dists)

    all_dists = torch.cat(all_dists)
    return all_dists.mean().item(), all_dists.std().item()
