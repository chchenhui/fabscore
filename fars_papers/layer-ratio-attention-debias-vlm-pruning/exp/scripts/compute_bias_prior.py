# Compute D2Pruner's offline positional bias prior for InternVL2.5-8B.
# Averages text->vision attention over N COCO images with a generic prompt
# at a target layer, producing the A_bias vector used for attention debiasing.
# Output: a .pt file containing the averaged attention tensor.

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
GENERIC_PROMPT = "Please describe the provided image."

BASE_DIR = Path(__file__).resolve().parent.parent


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
    return processed


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


def patch_attention_layer(model, target_layer):
    internlm2_mod = None
    for name, mod in sys.modules.items():
        if "modeling_internlm2" in name and hasattr(mod, "apply_rotary_pos_emb"):
            internlm2_mod = mod
            break
    if internlm2_mod is None:
        raise ImportError("modeling_internlm2 not found")

    apply_rotary_pos_emb = internlm2_mod.apply_rotary_pos_emb
    repeat_kv = internlm2_mod.repeat_kv

    layers = model.language_model.model.layers
    attn_module = layers[target_layer].attention

    def patched_forward(self, hidden_states, attention_mask=None,
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
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
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

    attn_module.forward = patched_forward.__get__(attn_module, type(attn_module))
    print(f"Patched layer {target_layer} for eager attention extraction")


def get_conv_template_fn():
    for name, mod in sys.modules.items():
        if "conversation" in name and hasattr(mod, "get_conv_template"):
            return mod.get_conv_template
    raise ImportError("conversation module not found")


def build_input(model, tokenizer, image_path):
    get_conv_template = get_conv_template_fn()
    image = Image.open(image_path).convert("RGB")
    max_num = getattr(model.config, "max_dynamic_patch", 12)
    force_size = getattr(model.config, "force_image_size", 448)
    use_thumb = getattr(model.config, "use_thumbnail", True)
    images = dynamic_preprocess(image, max_num=max_num, image_size=force_size, use_thumbnail=use_thumb)
    transform = build_transform(force_size)
    pixel_values = torch.stack([transform(img) for img in images]).to(torch.bfloat16).cuda()
    num_patches = pixel_values.shape[0]

    question = f"<image>\n{GENERIC_PROMPT}"
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
    query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].cuda()
    attention_mask = model_inputs["attention_mask"].cuda()

    flat = input_ids.reshape(-1)
    selected = (flat == img_context_token_id)
    vis_start = selected.nonzero(as_tuple=True)[0][0].item()
    vis_end = selected.nonzero(as_tuple=True)[0][-1].item() + 1

    vit_embeds = model.extract_feature(pixel_values)
    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(B * N, C)
    input_ids_flat = input_ids.reshape(B * N)
    sel = (input_ids_flat == img_context_token_id)
    input_embeds_flat[sel] = vit_embeds.reshape(-1, C).to(input_embeds_flat.device)
    input_embeds = input_embeds_flat.reshape(B, N, C)

    return input_embeds, attention_mask, vis_start, vis_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(BASE_DIR / "models" / "InternVL2_5-8B"))
    parser.add_argument("--target-layer", type=int, default=1,
                        help="Layer index to extract attention from (K-1 in D2Pruner notation)")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--coco-dir", type=str, default=str(BASE_DIR / "data" / "coco" / "train2014"))
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "data" / "bias_prior"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    image_files = sorted([f for f in os.listdir(args.coco_dir) if f.endswith(".jpg")])
    if len(image_files) < args.num_images:
        print(f"Warning: only {len(image_files)} images available, using all")
    else:
        image_files = random.sample(image_files, args.num_images)

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    print("Model loaded.")

    patch_attention_layer(model, args.target_layer)

    tokens_per_patch = model.num_image_token  # 256 for InternVL2.5-8B
    per_patch_accumulator = torch.zeros(tokens_per_patch, dtype=torch.float32)
    patch_count = 0
    layers = model.language_model.model.layers

    for img_file in tqdm(image_files, desc="Computing bias prior"):
        img_path = os.path.join(args.coco_dir, img_file)
        try:
            input_embeds, attention_mask, vis_start, vis_end = build_input(model, tokenizer, img_path)
            n_vis = vis_end - vis_start
            num_patches = n_vis // tokens_per_patch

            with torch.no_grad():
                model.language_model(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True,
                )

            stored_attn = layers[args.target_layer].attention._stored_attn_weights
            if stored_attn is None:
                continue

            vis_attn = stored_attn.mean(dim=1)[0, -1, vis_start:vis_end].float().cpu()

            for p in range(num_patches):
                start = p * tokens_per_patch
                end = start + tokens_per_patch
                per_patch_accumulator += vis_attn[start:end]
                patch_count += 1

            layers[args.target_layer].attention._stored_attn_weights = None

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    if patch_count > 0:
        avg_per_patch = per_patch_accumulator / patch_count
        out_path = os.path.join(args.output_dir, f"layer_{args.target_layer}.pt")
        torch.save(avg_per_patch, out_path)
        print(f"\nSaved per-patch bias prior to {out_path}")
        print(f"Averaged over {patch_count} patches from {len(image_files)} images")
        print(f"Shape: {avg_per_patch.shape}")
    else:
        print("No valid attention maps collected!")


if __name__ == "__main__":
    main()
