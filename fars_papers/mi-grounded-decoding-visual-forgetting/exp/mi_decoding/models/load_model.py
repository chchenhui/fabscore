# Shared model loading for Qwen2.5-VL family (VLAA-Thinker-7B, Qwen2.5-VL-7B-Instruct).
# Returns (model, processor) and provides input preparation helpers.
# VLAA-Thinker requires a thinking system prompt to trigger <think>/<answer> format.
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODELS = {
    "vlaa-thinker-7b": "UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B",
    "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
}

VLAA_THINKER_SYSTEM_PROMPT = (
    "You are VL-Thinking\U0001f914, a helpful assistant with excellent reasoning ability."
    " A user asks you a question, and you should try to solve it."
    " You should first think about the reasoning process in the mind and then provides the user with the answer."
    " The reasoning process and answer are enclosed within <think> </think> and"
    " <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>"
    " <answer> answer here </answer>"
)

VLAA_THINKER_IDS = [
    "UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B",
    "UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B",
]


def load_model(model_id, device_map="cuda"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def prepare_inputs(processor, image, question, model_device=None, system_prompt=None):
    content = [{"type": "text", "text": question}]
    if image is not None:
        content = [{"type": "image", "image": image}] + content

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if model_device is not None:
        inputs = inputs.to(model_device)
    return inputs
