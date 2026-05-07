# Sample 30 COCO images and define 5 prompt templates for Phase-0 diagnostic.
# Outputs data/phase0_samples.json with [{image_path, image_id, prompts}].
import json
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_DIR = os.path.join(BASE_DIR, "data", "coco", "train2014")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "phase0_samples.json")

PROMPTS = [
    "Describe the image.",
    "What do you see in this picture?",
    "Please describe the provided image.",
    "Tell me about this photo.",
    "What is shown in this image?",
]

NUM_IMAGES = 30
SEED = 42


def main():
    random.seed(SEED)
    all_images = sorted([f for f in os.listdir(COCO_DIR) if f.endswith(".jpg")])
    sampled = random.sample(all_images, NUM_IMAGES)

    samples = []
    for fname in sampled:
        image_id = fname.replace("COCO_train2014_", "").replace(".jpg", "")
        samples.append({
            "image_path": os.path.join("data", "coco", "train2014", fname),
            "image_id": image_id,
            "prompts": PROMPTS,
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Wrote {len(samples)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
