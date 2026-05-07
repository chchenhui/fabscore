# -*- coding: utf-8 -*-
"""
English word color gradient fusion - 8-letter words with colorful gradient effects
First 4 letters in one color, last 4 letters in another color, with gradient transition
"""

import os, re, json, argparse, string
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------- Font & rendering ----------------
SYS_FALLBACKS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont:
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)
    for p in SYS_FALLBACKS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def render_text_with_gradient(
    text: str,
    font: ImageFont.FreeTypeFont,
    canvas_size: Tuple[int, int],
    start_color: Tuple[int, int, int] = (255, 0, 0),  # Red
    end_color: Tuple[int, int, int] = (0, 255, 0),    # Green
    gradient_direction: str = "horizontal",  # "horizontal" or "vertical"
    padding_ratio: float = 0.02
) -> Image.Image:
    """Render text with gradient color effect."""
    # First render text in white on black to create a mask
    temp_canvas = Image.new("L", (1000, 1000), 0)
    temp_draw = ImageDraw.Draw(temp_canvas)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding_w = int(text_width * padding_ratio)
    padding_h = int(text_height * padding_ratio)
    actual_canvas_w = text_width + 2 * padding_w
    actual_canvas_h = text_height + 2 * padding_h

    final_canvas_w = min(canvas_size[0], actual_canvas_w)
    final_canvas_h = min(canvas_size[1], actual_canvas_h)

    # Create text mask
    mask_canvas = Image.new("L", (final_canvas_w, final_canvas_h), 0)
    mask_draw = ImageDraw.Draw(mask_canvas)
    
    x = (final_canvas_w - text_width) // 2 - bbox[0]
    y = (final_canvas_h - text_height) // 2 - bbox[1]
    
    mask_draw.text((x, y), text, fill=255, font=font)
    
    # Create gradient background
    gradient_canvas = Image.new("RGB", (final_canvas_w, final_canvas_h), (255, 255, 255))
    
    if gradient_direction == "horizontal":
        # Horizontal gradient
        for i in range(final_canvas_w):
            ratio = i / (final_canvas_w - 1) if final_canvas_w > 1 else 0
            r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
            g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
            b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
            
            for j in range(final_canvas_h):
                gradient_canvas.putpixel((i, j), (r, g, b))
    else:
        # Vertical gradient
        for j in range(final_canvas_h):
            ratio = j / (final_canvas_h - 1) if final_canvas_h > 1 else 0
            r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
            g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
            b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
            
            for i in range(final_canvas_w):
                gradient_canvas.putpixel((i, j), (r, g, b))
    
    # Apply text mask to gradient
    mask_array = np.array(mask_canvas)
    gradient_array = np.array(gradient_canvas)
    
    # Create final image
    final_canvas = Image.new("RGB", (final_canvas_w, final_canvas_h), (255, 255, 255))
    final_array = np.array(final_canvas)
    
    # Where mask is white (255), use gradient color; where mask is black (0), use white background
    mask_normalized = mask_array / 255.0
    for i in range(3):  # RGB channels
        final_array[:, :, i] = (gradient_array[:, :, i] * mask_normalized + 
                               255 * (1 - mask_normalized)).astype(np.uint8)
    
    return Image.fromarray(final_array)

def create_dual_gradient_overlay(
    img1: Image.Image, 
    img2: Image.Image, 
    canvas_size: Tuple[int, int] = (800, 400)
) -> Image.Image:
    """
    Create an overlay of two gradient images using blending for overlaps
    """
    # Create canvas
    canvas = Image.new("RGB", canvas_size, (255, 255, 255))
    canvas_array = np.array(canvas)
    
    w1, h1 = img1.size
    w2, h2 = img2.size
    
    # Position both images centered (no offset)
    x1 = (canvas_size[0] - w1) // 2
    y1 = (canvas_size[1] - h1) // 2
    
    # Position second image at same location (no offset)
    x2 = (canvas_size[0] - w2) // 2
    y2 = (canvas_size[1] - h2) // 2
    
    # Convert images to arrays
    img1_array = np.array(img1.convert("RGB"))
    img2_array = np.array(img2.convert("RGB"))
    
    # Create masks for non-white pixels
    mask1 = ~((img1_array[:, :, 0] > 240) & (img1_array[:, :, 1] > 240) & (img1_array[:, :, 2] > 240))
    mask2 = ~((img2_array[:, :, 0] > 240) & (img2_array[:, :, 1] > 240) & (img2_array[:, :, 2] > 240))
    
    # Place first image
    y1_end = min(y1 + h1, canvas_size[1])
    x1_end = min(x1 + w1, canvas_size[0])
    y1_start = max(y1, 0)
    x1_start = max(x1, 0)
    
    img1_y_start = max(0, -y1)
    img1_x_start = max(0, -x1)
    img1_y_end = img1_y_start + (y1_end - y1_start)
    img1_x_end = img1_x_start + (x1_end - x1_start)
    
    if y1_start < y1_end and x1_start < x1_end:
        canvas_region1 = canvas_array[y1_start:y1_end, x1_start:x1_end]
        img1_region = img1_array[img1_y_start:img1_y_end, img1_x_start:img1_x_end]
        mask1_region = mask1[img1_y_start:img1_y_end, img1_x_start:img1_x_end]
        
        # Apply first image where mask is true
        for c in range(3):
            canvas_region1[:, :, c] = np.where(mask1_region, img1_region[:, :, c], canvas_region1[:, :, c])
        
        canvas_array[y1_start:y1_end, x1_start:x1_end] = canvas_region1
    
    # Place second image with blending in overlap areas
    y2_end = min(y2 + h2, canvas_size[1])
    x2_end = min(x2 + w2, canvas_size[0])
    y2_start = max(y2, 0)
    x2_start = max(x2, 0)
    
    img2_y_start = max(0, -y2)
    img2_x_start = max(0, -x2)
    img2_y_end = img2_y_start + (y2_end - y2_start)
    img2_x_end = img2_x_start + (x2_end - x2_start)
    
    if y2_start < y2_end and x2_start < x2_end:
        canvas_region2 = canvas_array[y2_start:y2_end, x2_start:x2_end]
        img2_region = img2_array[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
        mask2_region = mask2[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
        
        # Check for overlap with first image
        # Create overlap detection mask
        is_white = (canvas_region2[:, :, 0] > 240) & (canvas_region2[:, :, 1] > 240) & (canvas_region2[:, :, 2] > 240)
        has_content = ~is_white  # Areas where first image has content
        
        # In overlap areas (where both images have content), blend 50:50
        overlap = has_content & mask2_region
        no_overlap = mask2_region & ~has_content
        
        for c in range(3):
            # Blend in overlap areas
            canvas_region2[:, :, c] = np.where(
                overlap,
                (canvas_region2[:, :, c].astype(np.float32) + img2_region[:, :, c].astype(np.float32)) / 2,
                canvas_region2[:, :, c]
            ).astype(np.uint8)
            
            # Direct placement in non-overlap areas
            canvas_region2[:, :, c] = np.where(no_overlap, img2_region[:, :, c], canvas_region2[:, :, c])
        
        canvas_array[y2_start:y2_end, x2_start:x2_end] = canvas_region2
    
    return Image.fromarray(canvas_array.astype(np.uint8))

# Predefined color schemes
COLOR_SCHEMES = {
    "fire": [(255, 0, 0), (255, 165, 0)],      # Red to Orange
    "ocean": [(0, 100, 200), (0, 200, 200)],   # Blue to Cyan  
    "forest": [(34, 139, 34), (154, 205, 50)], # Forest Green to Yellow Green
    "sunset": [(255, 94, 77), (255, 206, 84)], # Coral to Gold
    "purple": [(138, 43, 226), (221, 160, 221)], # Purple to Plum
    "rainbow1": [(255, 0, 0), (0, 255, 0)],    # Red to Green
    "rainbow2": [(0, 255, 0), (0, 0, 255)],    # Green to Blue
    "warm": [(255, 69, 0), (255, 215, 0)],     # Red Orange to Gold
    "cool": [(70, 130, 180), (176, 196, 222)], # Steel Blue to Light Steel Blue
    "neon": [(255, 20, 147), (0, 255, 255)],   # Deep Pink to Cyan
    "pure_red": [(255, 0, 0), (139, 0, 0)],    # Red to Dark Red
    "pure_green": [(0, 255, 0), (0, 139, 0)]   # Green to Dark Green
}

def fuse_word_8to1_gradient(
    word: str,
    font: ImageFont.FreeTypeFont,
    canvas_size: Tuple[int, int],
    color_scheme1: str = "fire",
    color_scheme2: str = "ocean", 
    gradient_direction: str = "horizontal"
):
    if len(word) != 8:
        raise ValueError(f"Word must be exactly 8 letters, got '{word}' ({len(word)}).")

    # Split into two groups
    group1 = word[:4]  # first 4 letters
    group2 = word[4:]  # last 4 letters

    # Get color schemes
    start_color1, end_color1 = COLOR_SCHEMES.get(color_scheme1, COLOR_SCHEMES["fire"])
    start_color2, end_color2 = COLOR_SCHEMES.get(color_scheme2, COLOR_SCHEMES["ocean"])

    # Render each group with gradient colors
    img1 = render_text_with_gradient(group1, font, canvas_size, start_color1, end_color1, gradient_direction)
    img2 = render_text_with_gradient(group2, font, canvas_size, start_color2, end_color2, gradient_direction)

    # Create overlay without offset
    fused = create_dual_gradient_overlay(img1, img2, canvas_size)

    return fused, [img1, img2], (group1, group2)

def slugify(s: str) -> str:
    base = re.sub(r"\s+", "", s)
    return re.sub(r"[^a-zA-Z0-9_-]", "", base.lower())

def compose_grid(images: List[Image.Image], rows: int, cols: int, gap: int = 12, bg=(255,255,255)) -> Image.Image:
    if not images:
        return Image.new("RGB", (100, 100), bg)

    assert len(images) == rows * cols
    W, H = images[0].width, images[0].height
    out = Image.new("RGB", (cols * W + (cols + 1) * gap, rows * H + (rows + 1) * gap), bg)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        x = gap + c * (W + gap)
        y = gap + r * (H + gap)
        out.paste(im.convert("RGB"), (x, y))
    return out

ALPHABET = set(string.ascii_letters)

def clean_word(word: str) -> str:
    return "".join(ch for ch in word if ch in ALPHABET).lower()

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="8->1 fusion with color gradient method.")
    ap.add_argument("--out_dir", default="color_gradient_fusion_out")
    ap.add_argument("--font", default=None, help="TTF/OTF path (optional).")
    ap.add_argument("--font_size", type=int, default=120, help="Font size for rendering text")
    ap.add_argument("--canvas_width", type=int, default=800, help="Canvas width")
    ap.add_argument("--canvas_height", type=int, default=400, help="Canvas height")
    ap.add_argument("--color_scheme1", choices=list(COLOR_SCHEMES.keys()), default="pure_red", help="Color scheme for first part")
    ap.add_argument("--color_scheme2", choices=list(COLOR_SCHEMES.keys()), default="pure_green", help="Color scheme for second part")
    ap.add_argument("--gradient_direction", choices=["horizontal", "vertical"], default="horizontal", help="Gradient direction")
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--save_originals", action="store_true")
    ap.add_argument("--words", nargs="*", help="8-letter words")
    ap.add_argument("--words_file", help="txt file, one word per line")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    labels_path = os.path.join(args.out_dir, "labels.jsonl")
    font = load_font(args.font, args.font_size)
    canvas_size = (args.canvas_width, args.canvas_height)

    words: List[str] = []
    if args.words: words.extend(args.words)
    if args.words_file and os.path.exists(args.words_file):
        with open(args.words_file, "r", encoding="utf-8") as f:
            words.extend([line.strip() for line in f if line.strip()])

    if not words:
        print("No words. Use --words or --words_file.")
        return

    with open(labels_path, "w", encoding="utf-8") as fw:
        image_counter = 1
        for raw in words:
            w = clean_word(raw)
            if len(w) != 8:
                print(f"[Skip] '{raw}' -> '{w}' ({len(w)}) needs 8 letters.")
                continue

            fused_img, orig_imgs, groups = fuse_word_8to1_gradient(
                w, font, canvas_size,
                color_scheme1=args.color_scheme1,
                color_scheme2=args.color_scheme2,
                gradient_direction=args.gradient_direction
            )

            # Save fused result with simple numeric name
            fused_name = f"{image_counter}.png"
            fused_path = os.path.join(args.out_dir, fused_name)
            fused_img.save(fused_path)

            orig_name = None
            if args.save_originals:
                orig_strip = compose_grid(orig_imgs, rows=1, cols=2, gap=args.gap)
                orig_name = f"{image_counter}_orig2parts.png"
                orig_path = os.path.join(args.out_dir, orig_name)
                orig_strip.save(orig_path)

            rec: Dict = {
                "filename": fused_name,
                "word": w,
                "rule": "color_gradient",
                "params": {
                    "color_scheme1": args.color_scheme1,
                    "color_scheme2": args.color_scheme2,
                    "gradient_direction": args.gradient_direction,
                    "canvas_size": canvas_size,
                    "font_size": args.font_size
                },
                "groups": f"{groups[0]}+{groups[1]}",
                "colors": f"{args.color_scheme1}+{args.color_scheme2}",
                "images": {"fused": fused_name, "original_parts": orig_name}
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Generated: {w} -> {fused_name}")
            image_counter += 1

    print(f"\nDone. Images + labels at: {args.out_dir}")
    print(f"Labels: {labels_path}")
    print(f"\nAvailable color schemes: {', '.join(COLOR_SCHEMES.keys())}")

if __name__ == "__main__":
    main()