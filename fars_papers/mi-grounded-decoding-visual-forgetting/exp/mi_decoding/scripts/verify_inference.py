# End-to-end inference verification: load VLAA-Thinker, process one MMStar
# sample with greedy decoding, and print the generated text.
import os
import sys
import torch
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

def main():
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from datasets import load_dataset

    model_id = "UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B"
    print(f"\nLoading model: {model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model and processor loaded.")

    print("\nLoading MMStar dataset...")
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    sample = ds[0]
    image = sample["image"]
    question = sample["question"]
    print(f"Question: {question}")
    print(f"Ground truth: {sample['answer']}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print(f"\nInput token count: {inputs['input_ids'].shape[1]}")
    print("Running greedy decoding (max_new_tokens=512)...")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)

    print(f"\nGenerated text ({len(generated_ids)} tokens):")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)

    print("\nINFERENCE VERIFICATION COMPLETE")

if __name__ == "__main__":
    main()
