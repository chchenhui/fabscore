# Utility metrics: mean per-token KL divergence between frozen LLM outputs
# on clean vs encrypted embeddings, computed over a held-out prompt set.

import torch
import torch.nn.functional as F
from key_search_bypass.encryptor.losses import _forward_from_embeds
from key_search_bypass.encryptor.model import KeyConditionedEncryptor


@torch.no_grad()
def compute_mean_kl(model, encryptor, input_ids, attention_mask, key_dim=128, batch_size=4):
    """Compute mean per-token KL divergence over the full eval set.

    Args:
        model: frozen LLM
        encryptor: trained KeyConditionedEncryptor
        input_ids: (N, T) token ids
        attention_mask: (N, T) attention mask
        key_dim: key dimension
        batch_size: eval batch size
    Returns:
        mean_kl: float
    """
    encryptor.eval()
    total_kl = 0.0
    total_tokens = 0
    n = input_ids.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = input_ids[start:end].cuda()
        mask = attention_mask[start:end].cuda()

        clean_embeds = model.model.embed_tokens(ids)
        k = KeyConditionedEncryptor.sample_key(ids.shape[0], key_dim, device=ids.device, dtype=clean_embeds.dtype)
        enc_embeds = encryptor(clean_embeds, k)

        clean_logits = _forward_from_embeds(model, clean_embeds, mask)
        enc_logits = _forward_from_embeds(model, enc_embeds, mask)

        clean_lp = F.log_softmax(clean_logits.float(), dim=-1)
        enc_lp = F.log_softmax(enc_logits.float(), dim=-1)
        kl = F.kl_div(enc_lp, clean_lp.exp(), reduction="none").sum(dim=-1)

        m = mask[:, :kl.shape[1]].float()
        total_kl += (kl * m).sum().item()
        total_tokens += m.sum().item()

    return total_kl / max(total_tokens, 1)
