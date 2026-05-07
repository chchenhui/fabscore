import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

local_model_dir = "llama-3-8B-Instruct"
# local_model_dir = "Llama-3-70B-Instruct"


input_dir = "folder_contain_clinical_notes"

output_dir = "llama_8b_deid"
# output_dir = "llama_70b_deid"

guideline_path = "guideline.txt"

os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,  
)

with open(guideline_path, "r", encoding="utf-8") as f:
    guideline = f.read()

for fname in os.listdir(input_dir):
    if not fname.endswith(".json"):
        continue

    fpath = os.path.join(input_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    note_text = data.get("text", "")

    messages = [
        {"role": "system", "content": "You are an experienced doctor who helps with PHI annotation."},
        {"role": "user", "content": guideline + "\n" + note_text}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.get_vocab() else None
    ]
    terminators = [tid for tid in terminators if tid is not None]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
        )

    response = outputs[0][input_ids.shape[-1]:]
    deid_output = tokenizer.decode(response, skip_special_tokens=True)

    out_data = {
        "filename": data.get("filename", fname),
        "original_text": note_text,
        "deid_output": deid_output
    }

    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"Finished {fname}, saved to {out_path}")
