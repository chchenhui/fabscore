# Utility, privacy, and diversity loss functions for encryptor training.
# Utility: KL divergence between frozen LLM outputs on clean vs encrypted embeddings.
# Privacy: cosine-similarity hinge loss penalizing |cos(h,z)| > eps.
# Diversity: pairwise L2 distance hinge loss between different-key encryptions.

import torch
import torch.nn.functional as F


def utility_loss(model, clean_embeds, encrypted_embeds, attention_mask):
    """KL divergence between LLM logits on clean vs encrypted embeddings.

    We need gradients to flow through encrypted_embeds only.
    clean_embeds forward is detached (no grad needed for the reference).

    Args:
        model: frozen Qwen2.5 model (model.model is the base, model.lm_head is the head)
        clean_embeds: (B, T, D) original token embeddings
        encrypted_embeds: (B, T, D) encrypted embeddings (requires grad)
        attention_mask: (B, T) attention mask
    Returns:
        scalar mean per-token KL divergence
    """
    with torch.no_grad():
        clean_out = _forward_from_embeds(model, clean_embeds, attention_mask)
    enc_out = _forward_from_embeds(model, encrypted_embeds, attention_mask)

    clean_logprobs = F.log_softmax(clean_out.float(), dim=-1)
    enc_logprobs = F.log_softmax(enc_out.float(), dim=-1)
    clean_probs = clean_logprobs.exp()

    kl = F.kl_div(enc_logprobs, clean_probs, reduction="none").sum(dim=-1)

    mask = attention_mask[:, :kl.shape[1]].float()
    kl_masked = (kl * mask).sum() / mask.sum().clamp(min=1.0)
    return kl_masked


def privacy_loss(clean_embeds, encrypted_embeds, attention_mask, eps=0.1):
    """Hinge loss penalizing |cos(h, z)| above margin eps.

    Args:
        clean_embeds: (B, T, D)
        encrypted_embeds: (B, T, D)
        attention_mask: (B, T)
        eps: margin threshold
    Returns:
        scalar mean per-token privacy loss
    """
    cos_sim = F.cosine_similarity(clean_embeds.float(), encrypted_embeds.float(), dim=-1)
    hinge = F.relu(cos_sim.abs() - eps)

    mask = attention_mask.float()
    loss = (hinge * mask).sum() / mask.sum().clamp(min=1.0)
    return loss


def diversity_loss(z1, z2, attention_mask, margin=2.0):
    """Hinge loss encouraging separation between encryptions with different keys.

    Args:
        z1: (B, T, D) encrypted with key k1
        z2: (B, T, D) encrypted with key k2
        attention_mask: (B, T)
        margin: minimum desired L2 distance
    Returns:
        scalar mean per-token diversity loss
    """
    l2_dist = torch.norm((z1 - z2).float(), dim=-1)
    hinge = F.relu(margin - l2_dist)

    mask = attention_mask.float()
    loss = (hinge * mask).sum() / mask.sum().clamp(min=1.0)
    return loss


def _forward_from_embeds(model, embeds, attention_mask):
    """Run LLM forward pass starting from embeddings (skipping embed_tokens).

    Uses model internals to feed pre-computed embeddings through the transformer.
    Returns logits of shape (B, T, V).
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    outputs = model.model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    hidden_states = outputs[0]
    logits = model.lm_head(hidden_states)
    return logits
