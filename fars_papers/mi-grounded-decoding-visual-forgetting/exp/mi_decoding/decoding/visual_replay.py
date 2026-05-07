# Visual replay decoding (VAPO Appendix A.5): two-pass strategy that re-inserts
# a downsampled image at 4 evenly-spaced punctuation boundaries in the reasoning
# trace to re-ground the model's visual attention.
import re
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from mi_decoding.models.load_model import VLAA_THINKER_SYSTEM_PROMPT


def _find_punctuation_positions(text):
    positions = []
    for m in re.finditer(r'[,.\n]', text):
        positions.append(m.end())
    return positions


def _snap_to_punctuation(target_pos, punct_positions, text_len):
    if not punct_positions:
        return target_pos
    best = min(punct_positions, key=lambda p: abs(p - target_pos))
    return best


def _compute_insertion_points(text, num_insertions=4):
    punct_positions = _find_punctuation_positions(text)
    text_len = len(text)
    if text_len == 0:
        return []
    segment_len = text_len / (num_insertions + 1)
    insertion_points = []
    for i in range(1, num_insertions + 1):
        target = int(i * segment_len)
        snapped = _snap_to_punctuation(target, punct_positions, text_len)
        if insertion_points and snapped <= insertion_points[-1]:
            snapped = min(target, text_len)
        insertion_points.append(snapped)
    return insertion_points


def _downsample_image(image, scale=0.5):
    w, h = image.size
    new_w = max(28, int(w * scale))
    new_h = max(28, int(h * scale))
    new_w = new_w - (new_w % 28)
    new_h = new_h - (new_h % 28)
    if new_w < 28:
        new_w = 28
    if new_h < 28:
        new_h = 28
    return image.resize((new_w, new_h), Image.LANCZOS)


def _build_interleaved_messages(system_prompt, question, original_image,
                                downsampled_image, text_segments):
    user_content = []
    if original_image is not None:
        user_content.append({"type": "image", "image": original_image})
    user_content.append({"type": "text", "text": question})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    assistant_content = []
    for i, seg in enumerate(text_segments):
        if i > 0:
            assistant_content.append({"type": "image", "image": downsampled_image})
        assistant_content.append({"type": "text", "text": seg})

    messages.append({"role": "assistant", "content": assistant_content})
    return messages


def generate_visual_replay(model, processor, image, question,
                           max_new_tokens=512, system_prompt=None,
                           num_insertions=4, downsample_scale=0.5):
    from mi_decoding.decoding.vanilla import generate_vanilla
    pass1_text = generate_vanilla(
        model, processor, image, question,
        max_new_tokens=max_new_tokens, system_prompt=system_prompt,
    )

    insertion_points = _compute_insertion_points(pass1_text, num_insertions)
    if not insertion_points:
        return pass1_text

    boundaries = [0] + insertion_points + [len(pass1_text)]
    text_segments = []
    for i in range(len(boundaries) - 1):
        seg = pass1_text[boundaries[i]:boundaries[i + 1]]
        text_segments.append(seg)

    if image is not None:
        ds_image = _downsample_image(image.copy(), scale=downsample_scale)
    else:
        return pass1_text

    prefill_segments = text_segments[:-1]
    messages = _build_interleaved_messages(
        system_prompt, question, image, ds_image, prefill_segments,
    )

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        continue_final_message=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated_ids = output_ids[0, input_len:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)

    prefill_text = pass1_text[:boundaries[-2]]
    full_text = prefill_text + generated_text
    return full_text
