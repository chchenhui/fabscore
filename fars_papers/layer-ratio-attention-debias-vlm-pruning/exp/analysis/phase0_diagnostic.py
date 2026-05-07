# Phase-0 go/no-go diagnostic for shallow-layer attention debiasing hypothesis.
# Extracts text->vision attention at shallow (1,2,3) and mid (2,4,8,12) layers of InternVL2.5-8B.
# Computes prompt stability, position correlation, and entropy metrics.
# Monkey-patches InternLM2FlashAttention2 for target layers to capture attention weights.
import argparse
import json
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from einops import rearrange
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "InternVL2_5-8B"
SAMPLES_PATH = BASE_DIR / "data" / "phase0_samples.json"
RESULTS_DIR = BASE_DIR / "results"
ATTN_MAPS_DIR = RESULTS_DIR / "phase0_attention_maps"

TARGET_LAYERS = [1, 2, 3, 4, 8, 12]
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def load_model(model_path):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True,
    ).eval().cuda()
    return model, tokenizer


def _get_internlm2_module():
    for name, mod in sys.modules.items():
        if "modeling_internlm2" in name and hasattr(mod, "apply_rotary_pos_emb"):
            return mod
    raise ImportError("modeling_internlm2 not found in sys.modules; model must be loaded first")


def patch_target_layers(model, target_layers):
    internlm2_mod = _get_internlm2_module()
    apply_rotary_pos_emb = internlm2_mod.apply_rotary_pos_emb
    repeat_kv = internlm2_mod.repeat_kv

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
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                causal_mask = torch.triu(
                    torch.full((q_len, kv_seq_len), float("-inf"), device=attn_weights.device, dtype=attn_weights.dtype),
                    diagonal=1
                )
                pad_mask = (1 - attention_mask[:, None, None, :].float()) * float("-inf")
                pad_mask = pad_mask.to(attn_weights.dtype)
                attn_weights = attn_weights + causal_mask[None, None, :, :] + pad_mask
            else:
                attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        self._stored_attn_weights = attn_weights.detach().cpu().float()

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.wo(attn_output)
        return attn_output, attn_weights, past_key_value

    layers = model.language_model.model.layers
    for layer_idx in target_layers:
        attn_module = layers[layer_idx].attention
        attn_module.forward = patched_forward.__get__(attn_module, type(attn_module))
        print(f"Patched layer {layer_idx} attention for eager computation")


def _get_conv_template_fn():
    for name, mod in sys.modules.items():
        if "conversation" in name and hasattr(mod, "get_conv_template"):
            return mod.get_conv_template
    raise ImportError("conversation module not found in sys.modules")


def build_input(model, tokenizer, image_path, prompt):
    get_conv_template = _get_conv_template_fn()
    pixel_values = load_image(image_path, model)
    num_patches = pixel_values.shape[0]

    question = f"<image>\n{prompt}"
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

    vit_embeds = model.extract_feature(pixel_values.cuda())

    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds_flat = input_embeds.reshape(B * N, C)
    input_ids_flat = input_ids.reshape(B * N)
    selected = (input_ids_flat == img_context_token_id)
    input_embeds_flat[selected] = vit_embeds.reshape(-1, C).to(input_embeds_flat.device)
    input_embeds = input_embeds_flat.reshape(B, N, C)

    vis_start = selected.nonzero(as_tuple=True)[0][0].item()
    vis_end = selected.nonzero(as_tuple=True)[0][-1].item() + 1
    num_vis_tokens = vis_end - vis_start

    return input_embeds, attention_mask, vis_start, vis_end, num_vis_tokens, num_patches


def load_image(image_path, model):
    from torchvision.transforms.functional import InterpolationMode
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size=448):
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set()
        for n in range(min_num, max_num + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i * j <= max_num and i * j >= min_num:
                        target_ratios.add((i, j))
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        best_ratio = (1, 1)
        best_ratio_diff = float("inf")
        area = orig_width * orig_height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio

        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    transform = build_transform(448)
    image = Image.open(str(image_path))
    images = dynamic_preprocess(
        image,
        min_num=model.config.min_dynamic_patch,
        max_num=model.config.max_dynamic_patch,
        image_size=model.config.force_image_size,
        use_thumbnail=model.config.use_thumbnail,
    )
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values.to(torch.bfloat16)


def extract_attention(model, input_embeds, attention_mask, vis_start, vis_end, target_layers):
    with torch.no_grad():
        outputs = model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )

    results = {}
    layers = model.language_model.model.layers
    for layer_idx in target_layers:
        attn = layers[layer_idx].attention._stored_attn_weights
        last_token_idx = input_embeds.shape[1] - 1
        attn_to_vis = attn[0, :, last_token_idx, vis_start:vis_end]
        attn_mean = attn_to_vis.mean(dim=0).numpy()
        results[layer_idx] = attn_mean
        layers[layer_idx].attention._stored_attn_weights = None

    return results


def compute_prompt_stability(attn_dict, num_images, num_prompts, target_layers):
    stability = {}
    for layer_idx in target_layers:
        sim_per_image = []
        for img_idx in range(num_images):
            vectors = []
            for p_idx in range(num_prompts):
                key = (img_idx, p_idx, layer_idx)
                if key in attn_dict:
                    vectors.append(attn_dict[key])
            if len(vectors) < 2:
                continue
            pairs = list(combinations(range(len(vectors)), 2))
            sims = []
            for i, j in pairs:
                v1 = vectors[i]
                v2 = vectors[j]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                sims.append(cos)
            sim_per_image.append(np.mean(sims))
        stability[layer_idx] = float(np.mean(sim_per_image)) if sim_per_image else 0.0
    return stability


def compute_position_correlation(attn_dict, num_images, num_prompts, target_layers, vis_token_info):
    corr = {}
    for layer_idx in target_layers:
        rho_vals = []
        for img_idx in range(num_images):
            for p_idx in range(num_prompts):
                key = (img_idx, p_idx, layer_idx)
                if key not in attn_dict:
                    continue
                attn_vec = attn_dict[key]
                n_tokens = len(attn_vec)
                info = vis_token_info.get((img_idx, p_idx))
                if info is None:
                    continue
                num_patches = info["num_patches"]
                tokens_per_patch = info["tokens_per_patch"]
                grid_h = int(math.sqrt(tokens_per_patch))
                grid_w = grid_h

                rhos_for_sample = []
                for patch_idx in range(num_patches):
                    start = patch_idx * tokens_per_patch
                    end = start + tokens_per_patch
                    if end > n_tokens:
                        break
                    patch_attn = attn_vec[start:end]

                    row_idx = np.arange(grid_h).repeat(grid_w)
                    col_idx = np.tile(np.arange(grid_w), grid_h)
                    dist_br = np.sqrt((row_idx - (grid_h - 1)) ** 2 + (col_idx - (grid_w - 1)) ** 2)

                    for pos_signal in [row_idx, col_idx, dist_br]:
                        r, _ = spearmanr(patch_attn, pos_signal)
                        if not np.isnan(r):
                            rhos_for_sample.append(abs(r))

                if rhos_for_sample:
                    rho_vals.append(np.mean(rhos_for_sample))

        corr[layer_idx] = float(np.mean(rho_vals)) if rho_vals else 0.0
    return corr


def compute_entropy(attn_dict, num_images, num_prompts, target_layers):
    entropy = {}
    for layer_idx in target_layers:
        ent_vals = []
        for img_idx in range(num_images):
            for p_idx in range(num_prompts):
                key = (img_idx, p_idx, layer_idx)
                if key not in attn_dict:
                    continue
                attn_vec = attn_dict[key]
                a_hat = attn_vec / (attn_vec.sum() + 1e-10)
                a_hat = np.clip(a_hat, 1e-10, None)
                h = -np.sum(a_hat * np.log(a_hat))
                ent_vals.append(h)
        entropy[layer_idx] = float(np.mean(ent_vals)) if ent_vals else 0.0
    return entropy


def verify_determinism(model, tokenizer, sample, target_layers):
    image_path = str(BASE_DIR / sample["image_path"])
    prompt = sample["prompts"][0]

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    input_embeds, attention_mask, vis_start, vis_end, _, _ = build_input(
        model, tokenizer, image_path, prompt
    )
    result1 = extract_attention(model, input_embeds, attention_mask, vis_start, vis_end, target_layers)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    input_embeds, attention_mask, vis_start, vis_end, _, _ = build_input(
        model, tokenizer, image_path, prompt
    )
    result2 = extract_attention(model, input_embeds, attention_mask, vis_start, vis_end, target_layers)

    for layer_idx in target_layers:
        diff = np.max(np.abs(result1[layer_idx] - result2[layer_idx]))
        print(f"  Layer {layer_idx}: max abs diff = {diff:.2e}")
        assert diff < 1e-5, f"Determinism check failed at layer {layer_idx}: max diff = {diff}"
    print("Determinism check PASSED")


def plot_attention_maps(attn_dict, vis_token_info, samples, target_layers, output_dir, num_examples=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_idx in range(min(num_examples, len(samples))):
        info = vis_token_info.get((img_idx, 0))
        if info is None:
            continue
        tokens_per_patch = info["tokens_per_patch"]
        grid_h = int(math.sqrt(tokens_per_patch))
        grid_w = grid_h
        num_patches = info["num_patches"]

        fig, axes = plt.subplots(1, len(target_layers), figsize=(4 * len(target_layers), 4))
        if len(target_layers) == 1:
            axes = [axes]

        image_id = samples[img_idx]["image_id"]
        for col, layer_idx in enumerate(target_layers):
            key = (img_idx, 0, layer_idx)
            if key not in attn_dict:
                continue
            attn_vec = attn_dict[key]
            first_patch = attn_vec[:tokens_per_patch]
            attn_map = first_patch.reshape(grid_h, grid_w)

            axes[col].imshow(attn_map, cmap="hot", interpolation="nearest")
            axes[col].set_title(f"Layer {layer_idx}")
            axes[col].axis("off")

        fig.suptitle(f"Image {image_id} - Prompt 0 (first patch)", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f"attn_map_img{img_idx}_{image_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved attention maps to {output_dir}")


def go_no_go_decision(stability, position_corr, entropy, target_layers):
    shallow_candidates = [1, 2, 3]
    mid_candidates = [4, 8, 12]

    best_ks = None
    best_score = -1
    for l in shallow_candidates:
        if l in stability and l in position_corr:
            score = stability[l] * position_corr[l]
            if score > best_score:
                best_score = score
                best_ks = l

    best_km = None
    best_km_score = -1
    for l in mid_candidates:
        if l in stability and l in position_corr:
            score = stability[l] * position_corr[l]
            if score > best_km_score:
                best_km_score = score
                best_km = l

    ks_stable = stability.get(best_ks, 0) > 0.8
    ks_pos_corr = position_corr.get(best_ks, 0) > 0.3
    km_lower_stability = stability.get(best_km, 1) < stability.get(best_ks, 0)
    km_lower_pos = position_corr.get(best_km, 1) < position_corr.get(best_ks, 0)

    if ks_stable and ks_pos_corr and (km_lower_stability or km_lower_pos):
        decision = "GO"
    elif ks_stable and not ks_pos_corr:
        decision = "PIVOT"
    else:
        decision = "REFUTE"

    return {
        "decision": decision,
        "K_s": best_ks,
        "K_m": best_km,
        "K_s_stability": stability.get(best_ks, 0),
        "K_s_position_corr": position_corr.get(best_ks, 0),
        "K_s_entropy": entropy.get(best_ks, 0),
        "K_m_stability": stability.get(best_km, 0),
        "K_m_position_corr": position_corr.get(best_km, 0),
        "K_m_entropy": entropy.get(best_km, 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run with 1 image, 1 prompt only")
    args = parser.parse_args()

    import wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "layer-ratio-attention-debias-vlm-pruning"),
        name="phase0-diagnostic" + ("-debug" if args.debug else ""),
        config={
            "target_layers": TARGET_LAYERS,
            "num_images": 1 if args.debug else 30,
            "num_prompts": 1 if args.debug else 5,
            "model": "InternVL2.5-8B",
        },
    )

    sys.path.insert(0, str(BASE_DIR))

    print("Loading model...")
    model, tokenizer = load_model(MODEL_PATH)
    print(f"Model loaded. num_image_token={model.num_image_token}")

    print("Patching target layers for attention extraction...")
    patch_target_layers(model, TARGET_LAYERS)

    with open(SAMPLES_PATH) as f:
        samples = json.load(f)

    if args.debug:
        samples = samples[:1]
        for s in samples:
            s["prompts"] = s["prompts"][:1]

    num_images = len(samples)
    num_prompts = len(samples[0]["prompts"])
    print(f"Running diagnostic: {num_images} images x {num_prompts} prompts = {num_images * num_prompts} forward passes")

    print("\n--- Determinism check ---")
    verify_determinism(model, tokenizer, samples[0], TARGET_LAYERS)

    attn_dict = {}
    vis_token_info = {}
    total = num_images * num_prompts
    count = 0

    for img_idx, sample in enumerate(samples):
        image_path = str(BASE_DIR / sample["image_path"])
        for p_idx, prompt in enumerate(sample["prompts"]):
            count += 1
            t0 = time.time()
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            input_embeds, attention_mask, vis_start, vis_end, num_vis_tokens, num_patches = build_input(
                model, tokenizer, image_path, prompt
            )
            result = extract_attention(
                model, input_embeds, attention_mask, vis_start, vis_end, TARGET_LAYERS
            )
            for layer_idx in TARGET_LAYERS:
                attn_dict[(img_idx, p_idx, layer_idx)] = result[layer_idx]

            vis_token_info[(img_idx, p_idx)] = {
                "num_patches": num_patches,
                "tokens_per_patch": model.num_image_token,
                "num_vis_tokens": num_vis_tokens,
            }

            elapsed = time.time() - t0
            if count <= 3 or count % 10 == 0:
                print(f"  [{count}/{total}] img={sample['image_id']} prompt={p_idx} "
                      f"vis_tokens={num_vis_tokens} patches={num_patches} time={elapsed:.1f}s")

    print("\n--- Computing metrics ---")
    stability = compute_prompt_stability(attn_dict, num_images, num_prompts, TARGET_LAYERS)
    position_corr = compute_position_correlation(attn_dict, num_images, num_prompts, TARGET_LAYERS, vis_token_info)
    entropy = compute_entropy(attn_dict, num_images, num_prompts, TARGET_LAYERS)

    print(f"\n{'Layer':>6} | {'Stability':>10} | {'Pos Corr':>10} | {'Entropy':>10}")
    print("-" * 50)
    for l in TARGET_LAYERS:
        print(f"{l:>6} | {stability.get(l, 0):>10.4f} | {position_corr.get(l, 0):>10.4f} | {entropy.get(l, 0):>10.4f}")

    decision_info = go_no_go_decision(stability, position_corr, entropy, TARGET_LAYERS)
    print(f"\n--- Decision: {decision_info['decision']} ---")
    print(f"  K_s = {decision_info['K_s']}, K_m = {decision_info['K_m']}")

    report = {
        "metrics": {
            str(l): {
                "prompt_stability": stability.get(l, 0),
                "position_correlation": position_corr.get(l, 0),
                "entropy": entropy.get(l, 0),
            }
            for l in TARGET_LAYERS
        },
        "decision": decision_info,
        "config": {
            "num_images": num_images,
            "num_prompts": num_prompts,
            "target_layers": TARGET_LAYERS,
            "model": "InternVL2.5-8B",
            "data_source": "data/coco/train2014 (RefCOCO images, not val2014)",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "phase0_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {report_path}")

    print("\n--- Plotting attention maps ---")
    plot_attention_maps(attn_dict, vis_token_info, samples, TARGET_LAYERS, ATTN_MAPS_DIR)

    wandb_table = wandb.Table(columns=["layer", "prompt_stability", "position_correlation", "entropy"])
    for l in TARGET_LAYERS:
        wandb_table.add_data(l, stability.get(l, 0), position_corr.get(l, 0), entropy.get(l, 0))
    wandb.log({"diagnostic_metrics": wandb_table})

    wandb.log({
        "decision": decision_info["decision"],
        "K_s": decision_info["K_s"],
        "K_m": decision_info["K_m"],
        "K_s_stability": decision_info["K_s_stability"],
        "K_s_position_corr": decision_info["K_s_position_corr"],
        "K_m_stability": decision_info["K_m_stability"],
        "K_m_position_corr": decision_info["K_m_position_corr"],
    })

    attn_map_dir = Path(ATTN_MAPS_DIR)
    if attn_map_dir.exists():
        for img_file in sorted(attn_map_dir.glob("*.png")):
            wandb.log({f"attention_map/{img_file.stem}": wandb.Image(str(img_file))})

    artifact = wandb.Artifact("phase0_report", type="report")
    artifact.add_file(str(report_path))
    wandb.log_artifact(artifact)

    wandb.finish()
    print("\nPhase-0 diagnostic complete.")


if __name__ == "__main__":
    main()
