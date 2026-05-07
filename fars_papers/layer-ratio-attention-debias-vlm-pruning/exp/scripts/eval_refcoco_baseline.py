# Evaluate D2Pruner / FastV / No-pruning baselines on RefCOCO grounding benchmarks.
# Adapted for InternVL2.5-8B (InternLM2 backbone). Uses model.chat() for no-pruning,
# and manual generation (no KV cache) for pruning methods since KV cache sizes differ
# across layers after pruning.
# Methods: --method none (upper bound), fastv (raw attn top-k), d2pruner (debiased + MIS)

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

GROUNDING_PROMPTS = {
    "ref": "Please provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>",
    "plain": "Please provide the bounding box coordinate of the region this sentence describes: {}",
}
GROUNDING_PROMPT = GROUNDING_PROMPTS["ref"]

BBOX_PATTERN = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')

BASE_DIR = Path(__file__).resolve().parent.parent

DS_COLLECTIONS = {
    "refcoco_val": str(BASE_DIR / "data/refcoco/refcoco_val.jsonl"),
    "refcoco_testA": str(BASE_DIR / "data/refcoco/refcoco_testA.jsonl"),
    "refcoco_testB": str(BASE_DIR / "data/refcoco/refcoco_testB.jsonl"),
    "refcoco+_val": str(BASE_DIR / "data/refcoco/refcoco+_val.jsonl"),
    "refcoco+_testA": str(BASE_DIR / "data/refcoco/refcoco+_testA.jsonl"),
    "refcoco+_testB": str(BASE_DIR / "data/refcoco/refcoco+_testB.jsonl"),
    "refcocog_val": str(BASE_DIR / "data/refcoco/refcocog_val.jsonl"),
    "refcocog_test": str(BASE_DIR / "data/refcoco/refcocog_test.jsonl"),
}


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


def load_image(image_path, model, max_num=None, dynamic=True):
    image = Image.open(image_path).convert("RGB")
    if max_num is None:
        max_num = getattr(model.config, "max_dynamic_patch", 12)
    force_size = getattr(model.config, "force_image_size", 448)
    use_thumb = getattr(model.config, "use_thumbnail", True)
    if dynamic:
        images = dynamic_preprocess(image, max_num=max_num, image_size=force_size, use_thumbnail=use_thumb)
    else:
        images = [image.resize((force_size, force_size))]
    transform = build_transform(force_size)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values.to(torch.bfloat16)


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


def get_conv_template_fn():
    for name, mod in sys.modules.items():
        if "conversation" in name and hasattr(mod, "get_conv_template"):
            return mod.get_conv_template
    raise ImportError("conversation module not found")


def find_visual_token_range(input_ids, img_context_token_id):
    flat = input_ids.reshape(-1)
    selected = (flat == img_context_token_id)
    if selected.sum() == 0:
        return None, None, 0
    indices = selected.nonzero(as_tuple=True)[0]
    vis_start = indices[0].item()
    vis_end = indices[-1].item() + 1
    return vis_start, vis_end, vis_end - vis_start


def setup_pruning_for_sample(model, tokenizer, pixel_values, question):
    num_patches = pixel_values.shape[0]
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    get_conv_template = get_conv_template_fn()
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    q_with_img = f"<image>\n{question}"
    template.append_message(template.roles[0], q_with_img)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
    query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    vis_start, vis_end, n_vis = find_visual_token_range(input_ids, img_context_token_id)
    model._vis_token_info = (vis_start, vis_end) if vis_start is not None else None
    model._pruning_applied = False
    model._keep_indices = None


def apply_pruning_patch(model, method, args):
    if method == "none":
        return

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

    prune_layer = args.prune_layer
    keep_ratio = args.keep_ratio
    bias_prior_path = args.bias_prior_path
    pivot_ratio = args.pivot_ratio
    sim_threshold = args.sim_threshold
    spatial_weight = args.spatial_weight

    shallow_layer = getattr(args, 'shallow_layer', 3)
    debiasing_mode = getattr(args, 'debiasing_mode', 'ratio')
    combo_alpha = getattr(args, 'combo_alpha', 0.5)
    residual_beta = getattr(args, 'residual_beta', 0.5)

    bias_prior = None
    if method == "d2pruner" and bias_prior_path:
        bias_prior = torch.load(bias_prior_path, map_location="cpu")
        print(f"Loaded bias prior from {bias_prior_path}, shape: {bias_prior.shape}")

    select_diverse_tokens_with_pivots = None
    if method in ("d2pruner", "online"):
        sys.path.insert(0, str(BASE_DIR / "external" / "D2Pruner"))
        from graph import select_diverse_tokens_with_pivots

    attn_layer = prune_layer - 1
    shallow_attn_layer = shallow_layer - 1
    layers = model.language_model.model.layers
    attn_module = layers[attn_layer].attention

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

    attn_module.forward = patched_attn_forward.__get__(attn_module, type(attn_module))
    print(f"Patched layer {attn_layer} attention for eager SDPA (attention extraction)")

    if method == "online" and shallow_attn_layer != attn_layer:
        shallow_attn_module = layers[shallow_attn_layer].attention
        shallow_attn_module.forward = patched_attn_forward.__get__(shallow_attn_module, type(shallow_attn_module))
        print(f"Patched layer {shallow_attn_layer} attention for eager SDPA (shallow prior)")

    lm_model = model.language_model.model
    original_forward = lm_model.__class__.forward

    def pruning_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                        past_key_values=None, inputs_embeds=None, use_cache=None,
                        output_attentions=None, output_hidden_states=None, return_dict=None):
        from transformers.modeling_outputs import BaseModelOutputWithPast
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("Must specify input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=device
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        if self.config.attn_implementation == 'flash_attention_2':
            attn_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length + past_key_values_length),
                    dtype=torch.bool, device=inputs_embeds.device
                )
            attn_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        should_prune = (seq_length > 1 and not getattr(model, '_pruning_applied', False))

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if idx == prune_layer and should_prune:
                model._pruning_applied = True
                stored_attn = layers[attn_layer].attention._stored_attn_weights
                if stored_attn is not None:
                    vis_info = getattr(model, '_vis_token_info', None)
                    if vis_info is not None:
                        vis_start, vis_end = vis_info
                        n_vis = vis_end - vis_start
                        n_keep = max(1, int(n_vis * keep_ratio))

                        img_attn = stored_attn.mean(dim=1)[0, -1, vis_start:vis_end]

                        if method == "d2pruner":
                            if bias_prior is not None:
                                bp = bias_prior.to(img_attn.device)
                                if bp.dim() == 1 and bp.shape[0] < n_vis:
                                    n_tiles = (n_vis + bp.shape[0] - 1) // bp.shape[0]
                                    bp_vis = bp.repeat(n_tiles)[:n_vis]
                                elif bp.dim() == 1:
                                    bp_vis = bp[:n_vis]
                                else:
                                    bp_vis = bp.mean(dim=0)[:n_vis]
                                img_attn = img_attn / (bp_vis + 1e-7)

                            features = hidden_states[0, vis_start:vis_end, :]
                            keep_indices = select_diverse_tokens_with_pivots(
                                features, n_keep, img_attn,
                                similarity_threshold=sim_threshold,
                                pivot_ratio=pivot_ratio,
                                spatial_weight=spatial_weight,
                                l1_weight=0, div_weight=0,
                            )
                        elif method == "online":
                            shallow_stored = layers[shallow_attn_layer].attention._stored_attn_weights
                            if debiasing_mode == "raw_mis":
                                pass
                            elif shallow_stored is not None:
                                a_shallow = shallow_stored.mean(dim=1)[0, -1, vis_start:vis_end]
                                if debiasing_mode == "ratio":
                                    img_attn = img_attn / (a_shallow + 1e-7)
                                elif debiasing_mode == "subtract":
                                    img_attn = img_attn - a_shallow
                                elif debiasing_mode == "zscore":
                                    mu_s = a_shallow.mean()
                                    std_s = a_shallow.std() + 1e-7
                                    a_shallow_z = (a_shallow - mu_s) / std_s
                                    mu_m = img_attn.mean()
                                    std_m = img_attn.std() + 1e-7
                                    img_attn = (img_attn - mu_m) / std_m - a_shallow_z
                                elif debiasing_mode == "weighted_combo":
                                    def _minmax_norm(x):
                                        xmin = x.min()
                                        xmax = x.max()
                                        return (x - xmin) / (xmax - xmin + 1e-7)
                                    img_attn = combo_alpha * _minmax_norm(img_attn) + (1 - combo_alpha) * _minmax_norm(a_shallow)
                                elif debiasing_mode == "entropy_ratio":
                                    ent_mid = -(img_attn * (img_attn + 1e-10).log()).sum()
                                    ent_shallow = -(a_shallow * (a_shallow + 1e-10).log()).sum()
                                    temp = ent_mid / (ent_shallow + 1e-7)
                                    a_shallow_scaled = a_shallow.pow(1.0 / (temp + 1e-7))
                                    a_shallow_scaled = a_shallow_scaled / (a_shallow_scaled.sum() + 1e-7)
                                    img_attn = img_attn / (a_shallow_scaled + 1e-7)
                                elif debiasing_mode == "residual":
                                    mu_s = a_shallow.mean()
                                    img_attn = img_attn - residual_beta * (a_shallow - mu_s)
                            if shallow_stored is not None:
                                layers[shallow_attn_layer].attention._stored_attn_weights = None

                            features = hidden_states[0, vis_start:vis_end, :]
                            keep_indices = select_diverse_tokens_with_pivots(
                                features, n_keep, img_attn,
                                similarity_threshold=sim_threshold,
                                pivot_ratio=pivot_ratio,
                                spatial_weight=spatial_weight,
                                l1_weight=0, div_weight=0,
                            )
                        else:
                            _, keep_indices = img_attn.topk(n_keep)

                        keep_indices_abs = keep_indices + vis_start
                        pre_vis = torch.arange(0, vis_start, device=hidden_states.device)
                        post_vis = torch.arange(vis_end, hidden_states.shape[1], device=hidden_states.device)
                        all_keep = torch.cat([pre_vis, keep_indices_abs.sort().values, post_vis])

                        model._keep_indices = all_keep

                        hidden_states = hidden_states[:, all_keep, :]
                        position_ids = position_ids[:, all_keep]

                        if attn_mask is not None:
                            if attn_mask.dim() == 4:
                                attn_mask = attn_mask[:, :, all_keep][:, :, :, all_keep]
                            elif attn_mask.dim() == 2:
                                attn_mask = attn_mask[:, all_keep]

                    layers[attn_layer].attention._stored_attn_weights = None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    lm_model.__class__ = type(
        "PrunedInternLM2Model",
        (lm_model.__class__,),
        {"forward": pruning_forward},
    )
    print(f"Patched InternLM2Model.forward for {method} pruning at layer {prune_layer}")


def compute_iou(pred_bbox, gt_bbox, img_w, img_h):
    if pred_bbox is None:
        return 0.0
    x1, y1, x2, y2 = pred_bbox
    if x1 + y1 + x2 + y2 >= 4:
        x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
    px1, py1, px2, py2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
    gx1, gy1, gx2, gy2 = gt_bbox

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    pred_area = max(0, px2 - px1) * max(0, py2 - py1)
    gt_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union_area = pred_area + gt_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def parse_bbox(text):
    matches = re.findall(BBOX_PATTERN, text)
    if matches:
        try:
            return (float(matches[0][0]), float(matches[0][1]),
                    float(matches[0][2]), float(matches[0][3]))
        except (ValueError, IndexError):
            return None
    return None


def evaluate_dataset(model, tokenizer, ds_name, ds_path, args):
    lines = open(ds_path).readlines()
    if args.max_samples > 0:
        lines = lines[:args.max_samples]

    if args.num_gpus > 1:
        rank = args.rank
        total = args.num_gpus
        lines = lines[rank::total]

    correct = 0
    total_count = 0
    outputs_list = []
    generation_config = dict(max_new_tokens=100, do_sample=False)

    for line in tqdm(lines, desc=f"{ds_name} (rank {getattr(args, 'rank', 0)})"):
        data = json.loads(line.strip())
        image_rel = data["image"]
        image_path = str(BASE_DIR / image_rel)
        text = data["sent"]
        gt_bbox = data["bbox"]
        img_w, img_h = data["width"], data["height"]

        try:
            pixel_values = load_image(image_path, model, max_num=args.max_num, dynamic=args.dynamic).cuda()
            prompt = GROUNDING_PROMPT.format(text)

            if args.method != "none":
                setup_pruning_for_sample(model, tokenizer, pixel_values, prompt)

            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing {image_path}: {e}")
            response = ""

        pred_bbox = parse_bbox(response)
        iou = compute_iou(pred_bbox, gt_bbox, img_w, img_h)
        total_count += 1
        if iou >= 0.5:
            correct += 1

        outputs_list.append({
            "answer": response,
            "gt_bbox": gt_bbox,
            "pred_bbox": list(pred_bbox) if pred_bbox else None,
            "iou": iou,
            "hw": (img_h, img_w),
        })

        if total_count <= 3:
            print(f"  [{total_count}] pred={pred_bbox}, gt={gt_bbox}, iou={iou:.3f}, response={repr(response[:100])}")

    return correct, total_count, outputs_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(BASE_DIR / "models" / "InternVL2_5-8B"))
    parser.add_argument("--datasets", type=str,
                        default="refcoco_val,refcoco_testA,refcoco_testB,"
                                "refcoco+_val,refcoco+_testA,refcoco+_testB,"
                                "refcocog_val,refcocog_test")
    parser.add_argument("--method", type=str, choices=["none", "fastv", "d2pruner", "online"], default="none")
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--prune-layer", type=int, default=2)
    parser.add_argument("--bias-prior-path", type=str, default="")
    parser.add_argument("--shallow-layer", type=int, default=3,
                        help="Shallow layer K_s for online debiasing (used with --method online)")
    parser.add_argument("--debiasing-mode", type=str, default="ratio",
                        choices=["ratio", "subtract", "zscore", "raw_mis",
                                 "weighted_combo", "entropy_ratio", "residual"],
                        help="Debiasing formula for online method")
    parser.add_argument("--combo-alpha", type=float, default=0.5,
                        help="Alpha for weighted_combo: score = alpha*norm(A_mid) + (1-alpha)*norm(A_shallow)")
    parser.add_argument("--residual-beta", type=float, default=0.5,
                        help="Beta for residual: score = A_mid - beta*(A_shallow - mean(A_shallow))")
    parser.add_argument("--pivot-ratio", type=float, default=0.7)
    parser.add_argument("--sim-threshold", type=float, default=0.8)
    parser.add_argument("--spatial-weight", type=float, default=0.5)
    parser.add_argument("--max-num", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--prompt-style", type=str, choices=["ref", "plain"], default="plain")
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "results" / "baseline"))
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()

    global GROUNDING_PROMPT
    GROUNDING_PROMPT = GROUNDING_PROMPTS[args.prompt_style]

    os.makedirs(args.out_dir, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    print(f"Model loaded. Method: {args.method}, keep_ratio: {args.keep_ratio}, prune_layer: {args.prune_layer}")
    print(f"  dynamic={args.dynamic}, max_num={args.max_num}, prompt_style={args.prompt_style}")

    apply_pruning_patch(model, args.method, args)

    all_results = {}
    for ds_name in datasets:
        if ds_name not in DS_COLLECTIONS:
            print(f"Unknown dataset: {ds_name}, skipping")
            continue
        ds_path = DS_COLLECTIONS[ds_name]
        if not os.path.exists(ds_path):
            print(f"Dataset file not found: {ds_path}, skipping")
            continue

        print(f"\nEvaluating {ds_name}...")
        correct, total_count, outputs = evaluate_dataset(model, tokenizer, ds_name, ds_path, args)
        acc = correct / total_count if total_count > 0 else 0.0
        print(f"{ds_name}: {correct}/{total_count} = {acc:.4f}")
        all_results[ds_name] = {"correct": correct, "total": total_count, "accuracy": acc}

        rank_suffix = f"_rank{args.rank}" if args.num_gpus > 1 else ""
        out_file = os.path.join(args.out_dir, f"{ds_name}{rank_suffix}.json")
        with open(out_file, "w") as f:
            json.dump({"correct": correct, "total": total_count, "accuracy": acc, "outputs": outputs}, f)
        print(f"Saved {ds_name} results to {out_file}")

    summary_file = os.path.join(args.out_dir, f"summary_rank{args.rank}.json" if args.num_gpus > 1 else "summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    if all_results:
        avg_acc = np.mean([r["accuracy"] for r in all_results.values()])
        print(f"\nAverage accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
