# Character Fusion Dataset Generator

A toolkit for generating synthetic text image datasets featuring character fusion and color gradient effects, supporting both Chinese characters and English words. For project "Visible Yet Unreadable: A Systematic Blind Spot of
Vision–Language Models Across Writing Systems". 

## 🌟 Features

- **Chinese Character Fusion**: Generate fused Chinese character images using various blending modes
- **English Color Gradient**: Create gradient-colored English word images with artistic effects
- **Multiple Fusion Modes**: Support for left-right split, top-bottom split, diagonal cut, checkerboard, stripes, and alpha blending
- **Flexible Input**: Process sentences, idioms, or custom character pairs
- **Batch Processing**: Generate multiple variations and layouts efficiently
- **JSON Annotations**: Comprehensive metadata for each generated image

## 🚀 Quick Start

### Prerequisites

```bash
pip install pillow numpy scipy tqdm
```

### Chinese Character Fusion

```bash
# Generate from idioms file
python cin-gen.py --font path/to/chinese/font.otf --idioms_file cin_100.txt --out_dir chinese_output

# Generate from custom sentences
python cin-gen.py --font path/to/chinese/font.otf --sentence "机器学习" "人工智能" --out_dir chinese_output

# Generate all modes for comparison
python cin-gen.py --font path/to/chinese/font.otf --idioms_file cin_100.txt --all_modes --out_dir chinese_output
```

### English Color Gradient

```bash
# Generate from word list
python en-color-gradient-fusion.py --words_file en_100.txt --out_dir english_output

# Generate with custom color schemes
python en-color-gradient-fusion.py --words computer security language --color_scheme1 fire --color_scheme2 ocean --out_dir english_output

# Save original parts for comparison
python en-color-gradient-fusion.py --words_file en_100.txt --save_originals --out_dir english_output
```

## 📊 Fusion Modes

### Chinese Character Fusion

| Mode | Description | Visual Effect |
|------|-------------|---------------|
| `lr` | Left-Right Split | First char on left, second on right |
| `tb` | Top-Bottom Split | First char on top, second on bottom |
| `diag` | Diagonal Cut | Diagonal line separation |
| `checker` | Checkerboard | Alternating square pattern |
| `vstripes` | Vertical Stripes | Vertical stripe pattern |
| `hstripes` | Horizontal Stripes | Horizontal stripe pattern |
| `alpha` | Alpha Blending | Transparent overlay effect |

### English Color Schemes

| Scheme | Colors | Effect |
|--------|---------|---------|
| `fire` | Red → Orange | Warm flame effect |
| `ocean` | Blue → Cyan | Cool water effect |
| `forest` | Green → Yellow-Green | Natural forest effect |
| `sunset` | Coral → Gold | Sunset gradient |
| `rainbow1` | Red → Green | Vibrant transition |
| `neon` | Pink → Cyan | Electric neon effect |

## 📁 Project Structure

```
├── cin-gen.py                    # Chinese character fusion generator
├── en-color-gradient-fusion.py   # English color gradient generator
├── cin_100.txt                   # Sample Chinese idioms (100 entries)
├── en_100.txt                     # Sample English words (100 entries)
└── README.md                      # This file
```

## 🎯 Use Cases

- **Computer Vision Research**: Generate synthetic datasets for text recognition tasks
- **Font Style Transfer**: Create training data for style transfer models
- **Data Augmentation**: Expand existing text datasets with artistic variations
- **Typography Research**: Study character fusion and color gradient effects
- **Educational Tools**: Demonstrate text processing and image generation techniques

## 📋 Command Line Options

### Chinese Generator (`cin-gen.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `--font` | Chinese font path (required) | - |
| `--out_dir` | Output directory | `heti_out` |
| `--cell_size` | Cell size in pixels | 400 |
| `--cols` | Grid columns | 2 |
| `--modes` | Fusion modes (comma-separated) | `lr,tb,diag` |
| `--repeat` | Random variations per sample | 2 |
| `--all_modes` | Generate all-mode layout | False |
| `--separate_modes` | Generate separate mode images | False |
| `--sentence` | Input sentences | - |
| `--idioms_file` | Idioms file path | - |
| `--pairs` | Custom character pairs | - |

### English Generator (`en-color-gradient-fusion.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `--font` | Font path (optional) | System default |
| `--out_dir` | Output directory | `color_gradient_fusion_out` |
| `--font_size` | Font size | 120 |
| `--canvas_width` | Canvas width | 800 |
| `--canvas_height` | Canvas height | 400 |
| `--color_scheme1` | First part color scheme | `pure_red` |
| `--color_scheme2` | Second part color scheme | `pure_green` |
| `--gradient_direction` | Gradient direction | `horizontal` |
| `--save_originals` | Save original parts | False |
| `--words` | Input words | - |
| `--words_file` | Words file path | - |

## 📈 Output Format

Both generators create:
- **Image files**: PNG format with artistic effects
- **Annotation file**: `labels.jsonl` with comprehensive metadata

### Sample Annotation Entry

```json
{
  "image": "example.png",
  "source": "idiom",
  "raw_text": "画蛇添足",
  "pairs": ["画+蛇", "添+足"],
  "modes": ["lr", "tb"],
  "grid": {"rows": 2, "cols": 1, "cell_size": 400, "gap": 14}
}
```

## 🔧 Advanced Usage

### Batch Processing Multiple Files

```bash
# Process multiple idiom files
for file in idioms_*.txt; do
    python cin-gen.py --font font.otf --idioms_file "$file" --out_dir "output_$(basename $file .txt)"
done
```

### Custom Font Integration

```bash
# Use different fonts for variety
python cin-gen.py --font NotoSerifSC-Regular.otf --sentence "传统字体"
python cin-gen.py --font SourceHanSansCN-Bold.otf --sentence "现代字体"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new fusion modes or color schemes
- Improve documentation
- Add support for new languages

## 📄 License

This project is open source. Please check individual file headers for specific license information.

## 🙏 Acknowledgments

- Built with Python PIL, NumPy, and SciPy
- Supports various font formats including TTF and OTF
- Inspired by artistic text effects and computer vision research

---

**Generated with ❤️ for text processing and computer vision research**
