#!/bin/bash
# Debug: print attention statistics for online method
cd /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp
source .venv/bin/activate
export HF_HOME=/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface
rm -rf /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/.cache/huggingface/modules/transformers_modules/InternVL2_5_hyphen_8B/__pycache__

python -c "
import sys, os, json, torch, math
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from einops import rearrange

BASE_DIR = Path('/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/layer-ratio-attention-debias-vlm-pruning/exp')
sys.path.insert(0, str(BASE_DIR / 'scripts'))
from eval_refcoco_baseline import (
    load_model, load_image, setup_pruning_for_sample,
    find_visual_token_range, DS_COLLECTIONS,
    IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
    get_conv_template_fn, GROUNDING_PROMPTS
)

model, tokenizer = load_model(str(BASE_DIR / 'models' / 'InternVL2_5-8B'))

# Patch layers 2 and 3 for attention extraction
sys.path.insert(0, str(BASE_DIR / 'scripts'))
internlm2_mod = None
for name, mod in sys.modules.items():
    if 'modeling_internlm2' in name and hasattr(mod, 'apply_rotary_pos_emb'):
        internlm2_mod = mod
        break
apply_rotary_pos_emb = internlm2_mod.apply_rotary_pos_emb
repeat_kv = internlm2_mod.repeat_kv

layers = model.language_model.model.layers

for layer_idx in [2, 3]:  # layers for K_s=3 and K_m=4
    attn_module = layers[layer_idx].attention
    def make_patched(orig_module):
        def patched_attn_forward(self, hidden_states, attention_mask=None,
                                  position_ids=None, past_key_value=None,
                                  output_attentions=False, use_cache=False, **kwargs):
            bsz, q_len, _ = hidden_states.size()
            qkv_states = self.wqkv(hidden_states)
            qkv_states = rearrange(qkv_states, 'b q (h gs d) -> b q h gs d',
                                   gs=2 + self.num_key_value_groups, d=self.head_dim)
            query_states = qkv_states[..., :self.num_key_value_groups, :]
            query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
            key_states = qkv_states[..., -2, :]
            value_states = qkv_states[..., -1, :]
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
            rope_seq_len = max(kv_seq_len, int(position_ids.max()) + 1) if position_ids is not None else kv_seq_len
            cos, sin = self.rotary_emb(value_states, seq_len=rope_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_kv_out = (key_states, value_states) if use_cache else None
            key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
            value_states_rep = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(torch.full((q_len, kv_seq_len), float('-inf'), device=attn_weights.device, dtype=attn_weights.dtype), diagonal=kv_seq_len - q_len + 1)
            attn_weights = attn_weights + causal_mask[None, None, :, :]
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    pad_mask = (1 - attention_mask[:, None, None, :].float()) * float('-inf')
                    pad_mask = pad_mask.to(attn_weights.dtype)
                    attn_weights = attn_weights + pad_mask
                elif attention_mask.dim() == 4:
                    attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            self._stored_attn_weights = attn_weights.detach()
            attn_output = torch.matmul(attn_weights, value_states_rep)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.wo(attn_output)
            return attn_output, attn_weights, past_kv_out
        return patched_attn_forward
    attn_module.forward = make_patched(attn_module).__get__(attn_module, type(attn_module))

# Load a sample image
ds_path = DS_COLLECTIONS['refcoco_val']
with open(ds_path) as f:
    lines = [json.loads(l) for l in f][:5]

for i, data in enumerate(lines):
    image_path = str(BASE_DIR / data['image'])
    text = data['sent']
    pixel_values = load_image(image_path, model, max_num=12, dynamic=True).cuda()
    
    num_patches = pixel_values.shape[0]
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    get_conv_template = get_conv_template_fn()
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    q_with_img = f'<image>\nPlease provide the bounding box coordinate of the region this sentence describes: <ref>{text}</ref>'
    template.append_message(template.roles[0], q_with_img)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
    query = query.replace('<image>', image_tokens, 1)
    model_inputs = tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].cuda()
    attention_mask = model_inputs['attention_mask'].cuda()
    
    vis_start, vis_end, n_vis = find_visual_token_range(input_ids, img_context_token_id)
    
    vit_embeds = model.extract_feature(pixel_values)
    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(B * N, C)
    input_ids_flat = input_ids.reshape(B * N)
    sel = (input_ids_flat == img_context_token_id)
    input_embeds_flat[sel] = vit_embeds.reshape(-1, C).to(input_embeds_flat.device)
    input_embeds = input_embeds_flat.reshape(B, N, C)
    
    with torch.no_grad():
        model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask,
                             output_attentions=False, use_cache=False, return_dict=True)
    
    for layer_idx in [2, 3]:
        stored = layers[layer_idx].attention._stored_attn_weights
        if stored is not None:
            attn_vis = stored.mean(dim=1)[0, -1, vis_start:vis_end].float().cpu()
            print(f'Sample {i}, Layer {layer_idx+1}: min={attn_vis.min():.8f}, max={attn_vis.max():.8f}, mean={attn_vis.mean():.8f}, std={attn_vis.std():.8f}, n_vis={n_vis}')
            layers[layer_idx].attention._stored_attn_weights = None
    
    # Compute the ratio
    l2_stored = None
    l3_stored = None
    with torch.no_grad():
        model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask,
                             output_attentions=False, use_cache=False, return_dict=True)
    l2_attn = layers[2].attention._stored_attn_weights.mean(dim=1)[0, -1, vis_start:vis_end].float().cpu()
    l3_attn = layers[3].attention._stored_attn_weights.mean(dim=1)[0, -1, vis_start:vis_end].float().cpu()
    ratio = l3_attn / (l2_attn + 1e-7)
    print(f'Sample {i}, Ratio A_mid/A_shallow: min={ratio.min():.4f}, max={ratio.max():.4f}, mean={ratio.mean():.4f}, std={ratio.std():.4f}')
    layers[2].attention._stored_attn_weights = None
    layers[3].attention._stored_attn_weights = None
    print()
"
