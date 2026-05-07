# Extract only the COCO train2014 images needed by RefCOCO from a partial zip.
# Falls back to downloading individual images from HuggingFace if zip is incomplete.

import json
import os
import glob
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REFCOCO_DIR = BASE_DIR / "data" / "refcoco"
COCO_DIR = BASE_DIR / "data" / "coco" / "train2014"
ZIP_PATH = BASE_DIR / "data" / "coco" / "train2014.zip"


def get_needed_images():
    images = set()
    for f in glob.glob(str(REFCOCO_DIR / "*.jsonl")):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                filename = os.path.basename(d["image"])
                images.add(filename)
    return images


def extract_from_zip(needed):
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    already = set(os.listdir(COCO_DIR))
    to_extract = needed - already
    print(f"Already extracted: {len(already)}, still need: {len(to_extract)}")

    if not to_extract:
        print("All images already present!")
        return set()

    if not ZIP_PATH.exists():
        print(f"Zip file not found: {ZIP_PATH}")
        return to_extract

    extracted = 0
    failed = set()
    try:
        with zipfile.ZipFile(str(ZIP_PATH), "r") as zf:
            namelist = set(zf.namelist())
            for fname in to_extract:
                zip_entry = f"train2014/{fname}"
                if zip_entry in namelist:
                    try:
                        data = zf.read(zip_entry)
                        with open(COCO_DIR / fname, "wb") as out:
                            out.write(data)
                        extracted += 1
                        if extracted % 500 == 0:
                            print(f"  Extracted {extracted} images...")
                    except Exception:
                        failed.add(fname)
                else:
                    failed.add(fname)
    except zipfile.BadZipFile:
        print("Zip file is incomplete/corrupt, extracting what we can...")
        try:
            with zipfile.ZipFile(str(ZIP_PATH), "r") as zf:
                namelist = set(zf.namelist())
                for fname in to_extract:
                    zip_entry = f"train2014/{fname}"
                    if zip_entry in namelist:
                        try:
                            data = zf.read(zip_entry)
                            with open(COCO_DIR / fname, "wb") as out:
                                out.write(data)
                            extracted += 1
                        except Exception:
                            failed.add(fname)
                    else:
                        failed.add(fname)
        except Exception as e:
            print(f"Failed to open zip: {e}")
            failed = to_extract

    print(f"Extracted {extracted} from zip, {len(failed)} missing")
    return failed


def download_missing(missing):
    if not missing:
        return
    print(f"Downloading {len(missing)} missing images from COCO dataset API...")
    import urllib.request
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    errors = 0
    for fname in sorted(missing):
        url = f"http://images.cocodataset.org/train2014/{fname}"
        dest = COCO_DIR / fname
        try:
            urllib.request.urlretrieve(url, str(dest))
            downloaded += 1
            if downloaded % 100 == 0:
                print(f"  Downloaded {downloaded}/{len(missing)}")
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error downloading {fname}: {e}")
    print(f"Downloaded {downloaded}, errors: {errors}")


def main():
    needed = get_needed_images()
    print(f"RefCOCO needs {len(needed)} unique COCO train2014 images")

    missing = extract_from_zip(needed)

    if missing:
        download_missing(missing)

    final_count = len(set(os.listdir(COCO_DIR)) & needed)
    print(f"\nFinal: {final_count}/{len(needed)} images available")
    if final_count == len(needed):
        print("All images ready!")
    else:
        print(f"WARNING: {len(needed) - final_count} images still missing")


if __name__ == "__main__":
    main()
