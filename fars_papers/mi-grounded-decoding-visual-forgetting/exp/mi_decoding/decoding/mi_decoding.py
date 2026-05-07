# Adaptive MI decoding (M3ID-style) for Qwen2.5-VL.
# Two forward passes per token: conditioned (with image) and masked (zeroed visual embeddings).
# Corrected logits: l_hat = l_c + 1[max p_c < alpha] * (1-gamma_t)/gamma_t * (l_c - l_u)
# where gamma_t = exp(-lambda*(t + t0)), t0 = small offset (default 0, NOT prompt length).
import math
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from mi_decoding.models.load_model import prepare_inputs

IMAGE_PAD_TOKEN_ID = 151655
EOS_TOKEN_IDS = {151645, 151643}


def _prepare_masked_inputs_embeds(model, input_ids):
    embeds = model.model.get_input_embeddings()(input_ids)
    mask = (input_ids == IMAGE_PAD_TOKEN_ID).unsqueeze(-1)
    embeds = embeds.masked_fill(mask, 0.0)
    return embeds


def generate_mi_decoding(model, processor, image, question,
                         max_new_tokens=512, system_prompt=None,
                         lam=0.005, alpha=0.3, t0=0, max_weight=5.0,
                         fixed_gamma=None):
    inputs = prepare_inputs(processor, image, question,
                            model_device=model.device, system_prompt=system_prompt)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    prompt_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]

    if t0 == "prompt_length":
        t0 = prompt_len

    cache_position = torch.arange(prompt_len, device=input_ids.device)

    position_ids, rope_deltas = model.model.get_rope_index(
        input_ids, image_grid_thw, None, attention_mask=attention_mask,
    )

    kv_c = DynamicCache()
    model.model.rope_deltas = None
    with torch.no_grad():
        out_c = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=None,
            past_key_values=kv_c,
            use_cache=True,
            cache_position=cache_position,
        )
    logits_c = out_c.logits[:, -1:, :]
    rope_c = model.model.rope_deltas

    masked_embeds = _prepare_masked_inputs_embeds(model, input_ids)
    kv_u = DynamicCache()
    model.model.rope_deltas = None
    with torch.no_grad():
        out_u = model(
            inputs_embeds=masked_embeds,
            attention_mask=attention_mask,
            pixel_values=None,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            past_key_values=kv_u,
            use_cache=True,
            cache_position=cache_position,
        )
    logits_u = out_u.logits[:, -1:, :]
    rope_u = model.model.rope_deltas
    if rope_u is None:
        rope_u = rope_c

    generated_ids = []

    for t in range(max_new_tokens):
        l_c = logits_c[:, -1, :]
        l_u = logits_u[:, -1, :]

        p_c = F.softmax(l_c, dim=-1)
        max_p = p_c.max(dim=-1).values.item()

        if max_p < alpha:
            if fixed_gamma is not None:
                gamma_t = fixed_gamma
            else:
                gamma_t = math.exp(-lam * (t + t0))
            gamma_t = max(gamma_t, 1e-8)
            weight = (1.0 - gamma_t) / gamma_t
            if max_weight > 0:
                weight = min(weight, max_weight)
            corrected = l_c + weight * (l_c - l_u)
        else:
            corrected = l_c

        next_token = corrected.argmax(dim=-1)
        token_id = next_token.item()
        generated_ids.append(token_id)

        if token_id in EOS_TOKEN_IDS:
            break

        next_input = next_token.unsqueeze(0)
        cache_pos = torch.tensor([prompt_len + t], device=input_ids.device)
        new_attn_mask = torch.ones(
            (batch_size, prompt_len + t + 1), dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        model.model.rope_deltas = rope_c
        with torch.no_grad():
            out_c = model(
                input_ids=next_input,
                attention_mask=new_attn_mask,
                past_key_values=kv_c,
                use_cache=True,
                cache_position=cache_pos,
            )
        logits_c = out_c.logits
        kv_c = out_c.past_key_values

        model.model.rope_deltas = rope_u
        with torch.no_grad():
            out_u = model(
                input_ids=next_input,
                attention_mask=new_attn_mask,
                past_key_values=kv_u,
                use_cache=True,
                cache_position=cache_pos,
            )
        logits_u = out_u.logits
        kv_u = out_u.past_key_values

    model.model.rope_deltas = rope_c
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)
    return generated_text


def generate_masked_only(model, processor, image, question,
                         max_new_tokens=512, system_prompt=None):
    inputs = prepare_inputs(processor, image, question,
                            model_device=model.device, system_prompt=system_prompt)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_grid_thw = inputs.get("image_grid_thw")
    t0 = input_ids.shape[1]
    batch_size = input_ids.shape[0]

    position_ids, rope_deltas = model.model.get_rope_index(
        input_ids, image_grid_thw, None, attention_mask=attention_mask,
    )

    masked_embeds = _prepare_masked_inputs_embeds(model, input_ids)
    kv = DynamicCache()
    model.model.rope_deltas = None
    cache_position = torch.arange(t0, device=input_ids.device)

    with torch.no_grad():
        out = model(
            inputs_embeds=masked_embeds,
            attention_mask=attention_mask,
            pixel_values=None,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            past_key_values=kv,
            use_cache=True,
            cache_position=cache_position,
        )
    logits = out.logits[:, -1:, :]
    rope_stored = model.model.rope_deltas
    if rope_stored is None:
        rope_stored = rope_deltas

    generated_ids = []

    for t in range(max_new_tokens):
        next_token = logits[:, -1, :].argmax(dim=-1)
        token_id = next_token.item()
        generated_ids.append(token_id)

        if token_id in EOS_TOKEN_IDS:
            break

        next_input = next_token.unsqueeze(0)
        cache_pos = torch.tensor([t0 + t], device=input_ids.device)
        new_attn_mask = torch.ones(
            (batch_size, t0 + t + 1), dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        model.model.rope_deltas = rope_stored
        with torch.no_grad():
            out = model(
                input_ids=next_input,
                attention_mask=new_attn_mask,
                past_key_values=kv,
                use_cache=True,
                cache_position=cache_pos,
            )
        logits = out.logits
        kv = out.past_key_values

    generated_text = processor.decode(generated_ids, skip_special_tokens=True)
    return generated_text
