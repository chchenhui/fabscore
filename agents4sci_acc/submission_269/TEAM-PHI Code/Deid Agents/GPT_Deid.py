import os
from openai import AzureOpenAI
import json

endpoint = "your_Azure_endpoint"

model_name = "gpt-35-turbo"
deployment = "gpt-35-turbo"

# model_name = "gpt-4o-mini"
# deployment = "gpt-4o-mini"

subscription_key = "your_Azure_subscription_key"
api_version = "your_api_version"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


guideline_path = "guideline.txt"
with open(guideline_path, "r", encoding="utf-8") as f:
    guideline = f.read()

input_dir = "folder_contain_clinical_notes"


# output_dir = "gpt4omini_deid"
output_dir = "gpt35_deid"


os.makedirs(output_dir, exist_ok=True)


for fname in os.listdir(input_dir):
    if not fname.endswith(".json"):
        continue

    fpath = os.path.join(input_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    note_text = data.get("text", "")

    print(f">>> Processing {fname} ...")


    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": "You are an experienced doctor who helps with PHI annotation.",
            },
            {
                "role": "user",
                "content": guideline + "\n" + note_text,
            },
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
    )

    output_text = response.choices[0].message.content


    out_data = {
        "filename": data.get("filename", fname),
        "original_text": note_text,
        "deid_output": output_text,
    }

    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"Finished {fname}，saved to {out_path}")
