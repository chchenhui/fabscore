# Download COCO train2014 images needed by RefCOCO from HuggingFace datasets.
# Uses justram/COCO2014-Images which has image_id (int) and image (PIL) columns.
# Maps image_id -> COCO_train2014_000000{id}.jpg filename.

import json
import os
import glob
import re
from pathlib import Path
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
REFCOCO_DIR = BASE_DIR / "data" / "refcoco"
COCO_DIR = BASE_DIR / "data" / "coco" / "train2014"


def get_needed_image_ids():
    needed_ids = set()
    needed_fnames = set()
    pattern = re.compile(r"COCO_train2014_(\d+)\.jpg")
    for f in glob.glob(str(REFCOCO_DIR / "*.jsonl")):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                fname = os.path.basename(d["image"])
                needed_fnames.add(fname)
                m = pattern.match(fname)
                if m:
                    needed_ids.add(int(m.group(1)))
    return needed_ids, needed_fnames


def main():
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    needed_ids, needed_fnames = get_needed_image_ids()
    already = set(os.listdir(COCO_DIR))
    to_download_fnames = needed_fnames - already
    to_download_ids = set()
    for fname in to_download_fnames:
        m = re.match(r"COCO_train2014_(\d+)\.jpg", fname)
        if m:
            to_download_ids.add(int(m.group(1)))

    print(f"Need {len(needed_fnames)} images, already have {len(already & needed_fnames)}, downloading {len(to_download_ids)}")

    if not to_download_ids:
        print("All images already present!")
        return

    print("Loading COCO2014-Images train split from HuggingFace (streaming)...")
    ds = load_dataset("justram/COCO2014-Images", split="train", streaming=True)

    saved = 0
    scanned = 0
    for row in ds:
        scanned += 1
        image_id = row.get("image_id")
        if image_id is not None and image_id in to_download_ids:
            img = row.get("image")
            if img is not None:
                fname = f"COCO_train2014_{image_id:012d}.jpg"
                img.save(str(COCO_DIR / fname), "JPEG")
                saved += 1
                to_download_ids.discard(image_id)
                if saved % 500 == 0:
                    print(f"  Saved {saved} images, scanned {scanned} rows, {len(to_download_ids)} remaining")

        if not to_download_ids:
            break

    print(f"\nDone: saved {saved} images from {scanned} scanned rows")
    final = len(set(os.listdir(COCO_DIR)) & needed_fnames)
    print(f"Final: {final}/{len(needed_fnames)} images available")
    if final < len(needed_fnames):
        print(f"WARNING: {len(needed_fnames) - final} images still missing")


if __name__ == "__main__":
    main()
