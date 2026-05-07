# GPU script: Extract attention maps + retained token indices for visualization.
# Patches layers 1 and 11 simultaneously for eager SDPA attention extraction.
# Computes FastV (top-k at L2), D2Pruner (debiased+MIS at L2), Ours (raw MIS at L12).

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "external" / "D2Pruner"))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
KEEP_RATIO = 0.1
PIVOT_RATIO = 0.7
SIM_THRESHOLD = 0.8
SPATIAL_WEIGHT = 0.5

GROUNDING_PROMPT = "Please provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>"


def build_transform(input_size=448):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = best_ratio[0] * image_size
    target_height = best_ratio[1] * image_size
    blocks = best_ratio[0] * best_ratio[1]
    resized = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        thumbnail = image.resize((image_size, image_size))
        processed.append(thumbnail)
    return processed, best_ratio


def load_image(image_path, model, max_num=None):
    image = Image.open(image_path).convert("RGB")
    if max_num is None:
        max_num = getattr(model.config, "max_dynamic_patch", 12)
    force_size = getattr(model.config, "force_image_size", 448)
    use_thumb = getattr(model.config, "use_thumbnail", True)
    images, best_ratio = dynamic_preprocess(image, max_num=max_num, image_size=force_size, use_thumbnail=use_thumb)
    transform = build_transform(force_size)
    pixel_values = torch.stack([transform(img) for img in images])
    n_tiles = best_ratio[0] * best_ratio[1]
    has_thumbnail = use_thumb and n_tiles > 1
    return pixel_values.to(torch.bfloat16), best_ratio, has_thumbnail


def load_model(model_path):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True,
    ).eval().cuda()
    return model, tokenizer


def find_visual_token_range(input_ids, img_context_token_id):
    flat = input_ids.reshape(-1)
    selected = (flat == img_context_token_id)
    if selected.sum() == 0:
        return None, None, 0
    indices = selected.nonzero(as_tuple=True)[0]
    vis_start = indices[0].item()
    vis_end = indices[-1].item() + 1
    return vis_start, vis_end, vis_end - vis_start


def get_conv_template_fn():
    for name, mod in sys.modules.items():
        if "conversation" in name and hasattr(mod, "get_conv_template"):
            return mod.get_conv_template
    raise ImportError("conversation module not found")


def patch_attention_layers(model, layer_indices):
    from einops import rearrange
    internlm2_mod = None
    for name, mod in sys.modules.items():
        if "modeling_internlm2" in name and hasattr(mod, "apply_rotary_pos_emb"):
            internlm2_mod = mod
            break
    if internlm2_mod is None:
        raise ImportError("modeling_internlm2 not found")

    apply_rotary_pos_emb = internlm2_mod.apply_rotary_pos_emb
    repeat_kv = internlm2_mod.repeat_kv

    def patched_attn_forward(self, hidden_states, attention_mask=None,
                             position_ids=None, past_key_value=None,
                             output_attentions=False, use_cache=False, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        qkv_states = self.wqkv(hidden_states)
        qkv_states = rearrange(
            qkv_states, "b q (h gs d) -> b q h gs d",
            gs=2 + self.num_key_value_groups, d=self.head_dim,
        )
        query_states = qkv_states[..., :self.num_key_value_groups, :]
        query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
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

        causal_mask = torch.triu(
            torch.full((q_len, kv_seq_len), float("-inf"), device=attn_weights.device, dtype=attn_weights.dtype),
            diagonal=kv_seq_len - q_len + 1,
        )
        attn_weights = attn_weights + causal_mask[None, None, :, :]

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (1 - attention_mask[:, None, None, :].float()) * float("-inf")
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

    layers = model.language_model.model.layers
    for li in layer_indices:
        attn_module = layers[li].attention
        attn_module.forward = patched_attn_forward.__get__(attn_module, type(attn_module))
        attn_module._stored_attn_weights = None
        print(f"  Patched layer {li} attention for eager SDPA")


def extract_hidden_states_at_layer(model, inputs_embeds, attention_mask, position_ids, target_layer):
    lm_model = model.language_model.model
    if lm_model.config.attn_implementation == 'flash_attention_2':
        attn_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        attn_mask = attention_mask

    hidden_states = inputs_embeds
    for idx, decoder_layer in enumerate(lm_model.layers):
        if idx > target_layer:
            break
        past_key_value = None
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = layer_outputs[0]
    return hidden_states


def process_example(model, tokenizer, example, bias_prior, out_dir, idx):
    from graph import select_diverse_tokens_with_pivots

    image_path = str(BASE_DIR / example["image"])
    sent = example["sent"]
    bbox = example["bbox"]
    width, height = example["width"], example["height"]

    print(f"\n--- Example {idx}: {sent} ---")
    print(f"  Image: {image_path}")

    pixel_values, tile_ratio, has_thumbnail = load_image(image_path, model)
    pixel_values = pixel_values.cuda()
    n_cols, n_rows = tile_ratio
    n_tiles = n_cols * n_rows
    print(f"  Tiles: {n_tiles} ({n_cols}x{n_rows}), thumbnail={has_thumbnail}")

    prompt = GROUNDING_PROMPT.format(sent)
    num_patches = pixel_values.shape[0]
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    get_conv_template = get_conv_template_fn()
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    q_with_img = f"<image>\n{prompt}"
    template.append_message(template.roles[0], q_with_img)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
    query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].cuda()
    attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids)).cuda()

    vis_start, vis_end, n_vis = find_visual_token_range(input_ids, img_context_token_id)
    print(f"  Visual tokens: {n_vis} (range [{vis_start}, {vis_end}))")

    vit_embeds = model.extract_feature(pixel_values)
    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)
    input_ids_flat = input_ids.reshape(B * N)
    selected = (input_ids_flat == img_context_token_id)
    input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.dtype)
    input_embeds = input_embeds.reshape(B, N, C)

    seq_length = input_ids.shape[1]
    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)

    layers = model.language_model.model.layers
    for li in [1, 11]:
        layers[li].attention._stored_attn_weights = None

    hidden_states = extract_hidden_states_at_layer(model, input_embeds, attention_mask, position_ids, 11)

    attn_L2 = layers[1].attention._stored_attn_weights
    attn_L12 = layers[11].attention._stored_attn_weights

    if attn_L2 is None or attn_L12 is None:
        print(f"  WARNING: attention not captured (L2={attn_L2 is not None}, L12={attn_L12 is not None})")
        return

    a_shallow = attn_L2.mean(dim=1)[0, -1, vis_start:vis_end].float().cpu()
    a_mid = attn_L12.mean(dim=1)[0, -1, vis_start:vis_end].float().cpu()
    a_debiased = (a_mid / (a_shallow + 1e-7))

    n_keep = max(1, int(n_vis * KEEP_RATIO))
    print(f"  n_keep: {n_keep}")

    _, fastv_indices = a_shallow.topk(n_keep)
    fastv_indices = fastv_indices.sort().values

    bp = bias_prior.to(a_shallow.device)
    if bp.shape[0] < n_vis:
        n_rep = (n_vis + bp.shape[0] - 1) // bp.shape[0]
        bp_vis = bp.repeat(n_rep)[:n_vis]
    else:
        bp_vis = bp[:n_vis]
    d2_attn = a_shallow / (bp_vis + 1e-7)

    features_L2 = hidden_states[0, vis_start:vis_end, :].float()

    d2_indices = select_diverse_tokens_with_pivots(
        features_L2.cuda(), n_keep, d2_attn.cuda(),
        similarity_threshold=SIM_THRESHOLD,
        pivot_ratio=PIVOT_RATIO,
        spatial_weight=SPATIAL_WEIGHT,
        l1_weight=0, div_weight=0,
    ).cpu()

    ours_indices = select_diverse_tokens_with_pivots(
        features_L2.cuda(), n_keep, a_mid.cuda(),
        similarity_threshold=SIM_THRESHOLD,
        pivot_ratio=PIVOT_RATIO,
        spatial_weight=SPATIAL_WEIGHT,
        l1_weight=0, div_weight=0,
    ).cpu()

    layers[1].attention._stored_attn_weights = None
    layers[11].attention._stored_attn_weights = None

    save_data = {
        "a_shallow": a_shallow.numpy(),
        "a_mid": a_mid.numpy(),
        "a_debiased": a_debiased.numpy(),
        "fastv_indices": fastv_indices.numpy(),
        "d2pruner_indices": d2_indices.numpy(),
        "ours_indices": ours_indices.numpy(),
        "tile_grid": (n_cols, n_rows),
        "has_thumbnail": has_thumbnail,
        "n_vis": n_vis,
        "image_path": example["image"],
        "sent": sent,
        "bbox": bbox,
        "img_size": (width, height),
    }

    out_path = out_dir / f"attn_data_{idx}.pt"
    torch.save(save_data, out_path)
    print(f"  Saved to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-json", type=str,
                        default=str(BASE_DIR / "results" / "visualizations" / "selected_examples.json"))
    parser.add_argument("--model-path", type=str,
                        default=str(BASE_DIR / "models" / "InternVL2_5-8B"))
    parser.add_argument("--bias-prior-path", type=str,
                        default=str(BASE_DIR / "data" / "bias_prior" / "layer_1.pt"))
    parser.add_argument("--out-dir", type=str,
                        default=str(BASE_DIR / "results" / "visualizations"))
    parser.add_argument("--max-examples", type=int, default=-1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.examples_json) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples from {args.examples_json}")

    if args.max_examples > 0:
        examples = examples[:args.max_examples]
        print(f"  Limited to {len(examples)} examples")

    bias_prior = torch.load(args.bias_prior_path, map_location="cpu").float()
    print(f"Loaded bias prior shape: {bias_prior.shape}")

    print("Loading model...")
    model, tokenizer = load_model(args.model_path)
    print("Model loaded.")

    patch_attention_layers(model, [1, 11])

    for idx, example in enumerate(examples):
        with torch.no_grad():
            process_example(model, tokenizer, example, bias_prior, out_dir, idx)

    print("\nDone!")


if __name__ == "__main__":
    main()
