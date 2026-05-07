# Download missing COCO train2014 images using parallel connections.
# Uses ThreadPoolExecutor to download many images concurrently.

import json
import os
import glob
import re
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent
REFCOCO_DIR = BASE_DIR / "data" / "refcoco"
COCO_DIR = BASE_DIR / "data" / "coco" / "train2014"
NUM_WORKERS = 32


def get_needed_filenames():
    fnames = set()
    for f in glob.glob(str(REFCOCO_DIR / "*.jsonl")):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                fnames.add(os.path.basename(d["image"]))
    return fnames


def download_one(fname):
    dest = COCO_DIR / fname
    if dest.exists():
        return fname, True, "exists"
    url = f"http://images.cocodataset.org/train2014/{fname}"
    try:
        urllib.request.urlretrieve(url, str(dest))
        return fname, True, "downloaded"
    except Exception as e:
        return fname, False, str(e)


def main():
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    needed = get_needed_filenames()
    already = set(f for f in os.listdir(COCO_DIR) if f in needed)
    to_download = sorted(needed - already)
    print(f"Need {len(needed)}, have {len(already)}, downloading {len(to_download)} with {NUM_WORKERS} threads")

    if not to_download:
        print("All done!")
        return

    success = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(download_one, f): f for f in to_download}
        for future in as_completed(futures):
            fname, ok, msg = future.result()
            if ok:
                success += 1
            else:
                fail += 1
                if fail <= 10:
                    print(f"  FAIL: {fname}: {msg}")
            total = success + fail
            if total % 500 == 0:
                print(f"  Progress: {total}/{len(to_download)} (success={success}, fail={fail})")

    final = len(set(os.listdir(COCO_DIR)) & needed)
    print(f"\nFinal: {final}/{len(needed)} images (downloaded={success}, failed={fail})")


if __name__ == "__main__":
    main()
