# -*- coding: utf-8 -*-
import os, re, json, math, random, argparse, hashlib, unicodedata
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

# --------------------------
# Basic functions: character rendering, binarization, fusion masks
# --------------------------
def render_char(ch: str, font: ImageFont.FreeTypeFont, size: int, pad_ratio=0.12) -> Image.Image:
    """Render a single Chinese character as a centered grayscale image (white background, black text)"""
    canvas = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(canvas)
    # Calculate suitable font size (with padding)
    pad = int(size * pad_ratio)
    # Rough binary search for font size
    L, R = 10, size * 2
    best = L
    while L <= R:
        mid = (L + R) // 2
        f = ImageFont.truetype(font.path, mid) if hasattr(font, "path") else font.font_variant(size=mid)
        bbox = draw.textbbox((0, 0), ch, font=f)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= size - 2 * pad and h <= size - 2 * pad:
            best = mid
            L = mid + 1
        else:
            R = mid - 1
    f = ImageFont.truetype(font.path, best) if hasattr(font, "path") else font.font_variant(size=best)
    bbox = draw.textbbox((0, 0), ch, font=f)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Consider baseline offset, use actual bbox position
    x = (size - w) // 2 - bbox[0]
    y = (size - h) // 2 - bbox[1]
    draw.text((x, y), ch, fill=0, font=f)
    return canvas

def to_mask(img_L: Image.Image, thresh: int = 200) -> np.ndarray:
    """Convert grayscale image to 0/1 mask: text as 1, background as 0"""
    arr = np.array(img_L)
    return (arr < thresh).astype(np.uint8)

def mask_to_img(mask: np.ndarray) -> Image.Image:
    """Convert 0/1 mask back to grayscale image (black text on white background)"""
    arr = 255 * (1 - mask)  # 1 -> 0(black), 0 -> 255(white)
    return Image.fromarray(arr.astype(np.uint8), mode="L")

def combine_masks(a: np.ndarray, b: np.ndarray, mode: str = "lr", rng=None) -> np.ndarray:
    """Fusion: combine two character masks A and B into one. mode:
       lr=left-right split, tb=top-bottom split, checker=checkerboard, diag=diagonal cut, vstripes=vertical stripes, hstripes=horizontal stripes, alpha=transparent overlay
    """
    H, W = a.shape
    if rng is None:
        rng = random.Random()
    m = np.zeros_like(a)
    if mode == "lr":
        cut = W // 2  # Split from center
        m[:, :cut] = a[:, :cut]
        m[:, cut:] = b[:, cut:]
    elif mode == "tb":
        cut = H // 2  # Split from center
        m[:cut, :] = a[:cut, :]
        m[cut:, :] = b[cut:, :]
    elif mode == "checker":
        sz = rng.choice([8, 12, 16, 20])
        for y in range(H):
            for x in range(W):
                if ((x//sz) + (y//sz)) % 2 == 0:
                    m[y, x] = a[y, x]
                else:
                    m[y, x] = b[y, x]
    elif mode == "diag":
        k = rng.uniform(0.6, 1.4)  # Random slope
        bias = rng.randint(-H//4, H//4)
        yy, xx = np.mgrid[0:H, 0:W]
        line = k * (xx - W/2) + H/2 + bias
        m[yy < line] = a[yy < line]
        m[yy >= line] = b[yy >= line]
    elif mode == "vstripes":
        w = rng.choice([6, 8, 10, 12])
        for x in range(W):
            if (x // w) % 2 == 0:
                m[:, x] = a[:, x]
            else:
                m[:, x] = b[:, x]
    elif mode == "hstripes":
        h = rng.choice([6, 8, 10, 12])
        for y in range(H):
            if (y // h) % 2 == 0:
                m[y, :] = a[y, :]
            else:
                m[y, :] = b[y, :]
    elif mode == "alpha":
        # Simple "union" + sparse erosion to create overlay effect
        m = np.clip(a + b, 0, 1)
        # Light random denoising
        if rng.random() < 0.5:
            kernel = rng.choice([1,2])
            from scipy.ndimage import binary_erosion
            m = binary_erosion(m.astype(bool), iterations=kernel).astype(np.uint8)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return m

# --------------------------
# Draw individual "fused character" blocks
# --------------------------
def fuse_two_chars(c1: str, c2: str, font: ImageFont.FreeTypeFont, size: int,
                   modes_pool: List[str], rng=None) -> Tuple[Image.Image, str]:
    if rng is None:
        rng = random.Random()
    img1 = render_char(c1, font, size)
    img2 = render_char(c2, font, size)
    m1, m2 = to_mask(img1), to_mask(img2)
    mode = rng.choice(modes_pool)
    fused = combine_masks(m1, m2, mode=mode, rng=rng)
    return mask_to_img(fused), mode

# --------------------------
# Compose grid and output images
# --------------------------
def compose_grid(images: List[Image.Image],
                 rows: int, cols: int, gap: int = 12,
                 bg=(255, 255, 255)) -> Image.Image:
    """Layout multiple blocks in rows x cols arrangement"""
    assert len(images) == rows * cols
    W = images[0].width
    H = images[0].height
    out = Image.new("RGB",
                    (cols * W + (cols + 1) * gap,
                     rows * H + (rows + 1) * gap),
                    bg)
    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        x = gap + c * (W + gap)
        y = gap + r * (H + gap)
        out.paste(im.convert("RGB"), (x, y))
    return out

# --------------------------
# Split text into groups of 2 characters
# --------------------------
_CN_PUNCT = r"[，。、「」；：？！、《》——…,.!?:;()\[\]{}<>“”‘’·—\s]"

def pairize_text(text: str) -> List[Tuple[str, str]]:
    # Remove punctuation and whitespace
    cleaned = re.sub(_CN_PUNCT, "", text)
    # Keep only CJK unified ideographic characters
    cleaned = "".join(ch for ch in cleaned if "CJK" in unicodedata.name(ch, ""))
    if len(cleaned) < 2:
        return []
    if len(cleaned) % 2 == 1:
        cleaned = cleaned[:-1]
    pairs = [(cleaned[i], cleaned[i+1]) for i in range(0, len(cleaned), 2)]
    return pairs

# --------------------------
# Task: generate one or more images from sentences/idioms/custom sequences
# --------------------------
DEFAULT_MODES = ["lr", "tb", "diag"]

def make_image_from_pairs(pairs: List[Tuple[str, str]],
                          font: ImageFont.FreeTypeFont,
                          cell_size: int = 400,
                          cols: int = 2,
                          gap: int = 14,
                          modes_pool=DEFAULT_MODES,
                          rng=None):
    if rng is None:
        rng = random.Random()
    blocks, used_modes = [], []
    for (a, b) in pairs:
        im, mode = fuse_two_chars(a, b, font, size=cell_size, modes_pool=modes_pool, rng=rng)
        blocks.append(im)
        used_modes.append(mode)
    rows = math.ceil(len(blocks) / cols)
    # Pad with blank blocks if insufficient (rare case)
    while len(blocks) < rows * cols:
        blocks.append(Image.new("L", (cell_size, cell_size), 255).convert("RGB"))
        used_modes.append("pad")
    grid = compose_grid(blocks, rows=rows, cols=cols)
    return grid, used_modes, rows, cols

def make_all_modes_image(pairs: List[Tuple[str, str]],
                         font: ImageFont.FreeTypeFont,
                         cell_size: int = 400,
                         gap: int = 14):
    """Generate images with all three modes for each character pair, arranged in rows"""
    all_blocks = []
    all_modes = []
    
    # Generate three modes for each character pair
    for (a, b) in pairs:
        row_blocks = []
        row_modes = []
        for mode in DEFAULT_MODES:  # ["lr", "tb", "diag"]
            im, _ = fuse_two_chars(a, b, font, size=cell_size, modes_pool=[mode], rng=None)
            row_blocks.append(im)
            row_modes.append(mode)
        all_blocks.extend(row_blocks)
        all_modes.extend(row_modes)
    
    # Calculate grid layout: 3 columns (three modes), rows = number of character pairs
    cols = 3
    rows = len(pairs)
    grid = compose_grid(all_blocks, rows=rows, cols=cols, gap=gap)
    return grid, all_modes, rows, cols

def slugify(s: str) -> str:
    base = re.sub(r"\s+", "", s)
    base = re.sub(_CN_PUNCT, "", base)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    return f"{base[:12]}_{h}"

# --------------------------
# Main workflow
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="heti_out", help="Output directory")
    ap.add_argument("--font", required=True, help="Chinese font path (e.g., NotoSerifSC-Regular.otf / SourceHanSansCN-Regular.otf)")
    ap.add_argument("--cell_size", type=int, default=400, help="Cell size (pixels)")
    ap.add_argument("--cols", type=int, default=2, help="Grid columns (usually 2)")
    ap.add_argument("--gap", type=int, default=14, help="Grid spacing")
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Available fusion modes, comma-separated")
    ap.add_argument("--repeat", type=int, default=2, help="How many random styles to generate for each sample")
    ap.add_argument("--all_modes", action="store_true", help="Generate combined images containing all three modes for each original text")
    ap.add_argument("--separate_modes", action="store_true", help="Generate three separate images for each original text, corresponding to three fusion modes")
    ap.add_argument("--seed", type=int, default=42)
    # Three input methods:
    ap.add_argument("--sentence", nargs="*", help="Directly provide Chinese sentences (multiple allowed)")
    ap.add_argument("--idioms_file", help="Four-character idiom txt file, one idiom per line")
    ap.add_argument("--pairs", nargs="*", help="Custom pairs, e.g.: 不 想 上 班 那 就 别 上 (even number of characters)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    labels_path = os.path.join(args.out_dir, "labels.jsonl")
    font = ImageFont.truetype(args.font, 40)  # Font size will be dynamically adjusted during rendering
    modes_pool = [m.strip() for m in args.modes.split(",") if m.strip()]
    rng = random.Random(args.seed)

    all_samples = []

    # 1) Sentences
    if args.sentence:
        for sent in args.sentence:
            pairs = pairize_text(sent)
            if not pairs:
                continue
            all_samples.append(("sentence", sent, pairs))

    # 2) Idiom file (each 4 characters -> [(1,2), (3,4)])
    if args.idioms_file and os.path.exists(args.idioms_file):
        with open(args.idioms_file, "r", encoding="utf-8") as f:
            for line in f:
                idiom = line.strip()
                if not idiom:
                    continue
                clean = re.sub(_CN_PUNCT, "", idiom)
                if len(clean) < 4:
                    continue
                clean = clean[:4]
                pairs = [(clean[0], clean[1]), (clean[2], clean[3])]
                all_samples.append(("idiom", idiom, pairs))

    # 3) Manual pairs
    if args.pairs and len(args.pairs) >= 2 and len(args.pairs) % 2 == 0:
        chars = "".join(args.pairs)
        pairs = [(chars[i], chars[i+1]) for i in range(0, len(chars), 2)]
        all_samples.append(("pairs", chars, pairs))

    if not all_samples:
        print("No available input. Please use --sentence / --idioms_file / --pairs to provide data.")
        return

    with open(labels_path, "w", encoding="utf-8") as fw:
        sample_index = 1  # Used for generating filename sequence numbers
        for src_type, raw_text, pairs in tqdm(all_samples, desc="Generating"):
            if args.all_modes:
                # Generate combined images containing all three modes
                grid, used_modes, rows, cols = make_all_modes_image(
                    pairs, font, cell_size=args.cell_size, gap=args.gap
                )
                fname = f"{slugify(raw_text)}_all_modes.png"
                fpath = os.path.join(args.out_dir, fname)
                grid.save(fpath)

                rec = {
                    "image": fname,
                    "source": src_type,
                    "raw_text": raw_text,
                    "pairs": ["{}+{}".format(a, b) for a, b in pairs],
                    "modes": used_modes,
                    "grid": {"rows": rows, "cols": cols, "cell_size": args.cell_size, "gap": args.gap},
                    "layout": "all_modes"  # Mark as all-modes layout
                }
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            elif args.separate_modes:
                # Generate three separate mode images
                for mode_idx, mode in enumerate(DEFAULT_MODES, 1):  # 1, 2, 3
                    grid, used_modes, rows, cols = make_image_from_pairs(
                        pairs, font, cell_size=args.cell_size, cols=args.cols, gap=args.gap,
                        modes_pool=[mode], rng=None
                    )
                    fname = f"{sample_index}-{mode_idx}.png"
                    fpath = os.path.join(args.out_dir, fname)
                    grid.save(fpath)

                    rec = {
                        "image": fname,
                        "source": src_type,
                        "raw_text": raw_text,
                        "pairs": ["{}+{}".format(a, b) for a, b in pairs],
                        "modes": used_modes,
                        "mode_name": mode,  # Add mode name
                        "grid": {"rows": rows, "cols": cols, "cell_size": args.cell_size, "gap": args.gap},
                        "layout": "separate_mode"
                    }
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sample_index += 1
            else:
                # Original random generation method
                for rep in range(args.repeat):
                    grid, used_modes, rows, cols = make_image_from_pairs(
                        pairs, font, cell_size=args.cell_size, cols=args.cols, gap=args.gap,
                        modes_pool=modes_pool, rng=rng
                    )
                    fname = f"{slugify(raw_text)}_{rep}.png"
                    fpath = os.path.join(args.out_dir, fname)
                    grid.save(fpath)

                    rec = {
                        "image": fname,
                        "source": src_type,
                        "raw_text": raw_text,          # Original sentence/idiom/character sequence
                        "pairs": ["{}+{}".format(a, b) for a, b in pairs],
                        "modes": used_modes,          # Mode corresponding to each fusion block
                        "grid": {"rows": rows, "cols": cols, "cell_size": args.cell_size, "gap": args.gap}
                    }
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Complete! Images and annotations output to: {args.out_dir}")
    print(f"Annotation file: {labels_path}")
    print("One JSON per line, containing image filename, original text, pairing method, fusion modes and grid information.")
    
if __name__ == "__main__":
    main()