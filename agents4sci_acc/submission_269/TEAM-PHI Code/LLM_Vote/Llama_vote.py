
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# local_model_dir = "llama-3-8B-Instruct"
local_model_dir = "Llama-3-70B-Instruct"


print(">>> Loading local llama model...")
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()


guideline_path = "vote_guideline_independent.txt"
# guideline_path = "vote_guideline_info.txt"


with open(guideline_path, "r", encoding="utf-8") as f:
    guideline = f.read()


llm_results = """
deid agents and evaluation agents' output
"""


messages = [
    {
        "role": "system",
        "content": "You are an experienced evaluation agent who helps with evaluate results of several de-identification (deid) agents.",
    },
    {
        "role": "user",
        "content": guideline + "\n" + llm_results,
    },
]


input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)


gen_kwargs = dict(
    max_new_tokens=1024,
    do_sample=True,  
    temperature=1.0,
    top_p=1.0,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
)


with torch.no_grad():
    outputs = model.generate(input_ids, **gen_kwargs)

response_tokens = outputs[0][input_ids.shape[-1]:]
output_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

print("=== LLaMA Evaluation Output ===")
print(output_text)
