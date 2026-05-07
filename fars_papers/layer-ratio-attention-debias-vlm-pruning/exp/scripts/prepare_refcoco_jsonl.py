# Convert RefCOCO/RefCOCO+/RefCOCOg from HuggingFace datasets to JSONL format
# expected by InternVL/D2Pruner evaluation scripts.
# Output: data/refcoco/{refcoco,refcoco+,refcocog}_{val,testA,testB,test}.jsonl

import json
import os
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "refcoco")
COCO_IMAGE_PREFIX = "data/coco/train2014"

DATASET_CONFIGS = {
    "refcoco": {
        "hf_name": "jxu124/refcoco",
        "splits": {
            "validation": "refcoco_val",
            "test": "refcoco_testA",
            "testB": "refcoco_testB",
        },
    },
    "refcoco+": {
        "hf_name": "jxu124/refcocoplus",
        "splits": {
            "validation": "refcoco+_val",
            "test": "refcoco+_testA",
            "testB": "refcoco+_testB",
        },
    },
    "refcocog": {
        "hf_name": "jxu124/refcocog",
        "splits": {
            "validation": "refcocog_val",
            "test": "refcocog_test",
        },
    },
}


def convert_dataset(hf_name, splits, output_dir):
    ds = load_dataset(hf_name)
    for split_key, output_name in splits.items():
        if split_key not in ds:
            print(f"  WARNING: split '{split_key}' not found in {hf_name}, skipping")
            continue

        split_data = ds[split_key]
        output_path = os.path.join(output_dir, f"{output_name}.jsonl")
        count = 0

        with open(output_path, "w") as f:
            for row in split_data:
                raw_image_info = json.loads(row["raw_image_info"])
                raw_anns = json.loads(row["raw_anns"])
                w = raw_image_info["width"]
                h = raw_image_info["height"]
                coco_filename = raw_image_info["file_name"]
                image_path = os.path.join(COCO_IMAGE_PREFIX, coco_filename)

                bbox_xywh = raw_anns["bbox"]
                x1 = bbox_xywh[0]
                y1 = bbox_xywh[1]
                x2 = x1 + bbox_xywh[2]
                y2 = y1 + bbox_xywh[3]
                bbox = [x1, y1, x2, y2]

                for sent_info in row["sentences"]:
                    entry = {
                        "image": image_path,
                        "sent": sent_info["sent"],
                        "bbox": bbox,
                        "width": w,
                        "height": h,
                    }
                    f.write(json.dumps(entry) + "\n")
                    count += 1

        print(f"  {output_name}: {count} entries -> {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"Processing {dataset_name}...")
        convert_dataset(config["hf_name"], config["splits"], OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
