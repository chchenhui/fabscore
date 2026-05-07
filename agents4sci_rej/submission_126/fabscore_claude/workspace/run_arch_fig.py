"""
Minimal reproduction of create_model_architecture_figure() from create_figures.py
Uses a placeholder image instead of the proprietary DogHeart dataset image.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# plt.style.use('seaborn-paper')  # not available in newer matplotlib

def create_model_architecture_figure(out_path):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_aspect('equal')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Placeholder image (simulating Input: Dog X-ray)
    img = np.random.rand(160, 200, 3)  # dummy placeholder
    img_width = 2.0
    img_height = img_width / 1.25
    ax.imshow(img, extent=[0.2, 0.2 + img_width, 3.5 - img_height/2, 3.5 + img_height/2])
    ax.text(0.2 + img_width/2, 3.5 + img_height/2 + 0.2, 'Input: Dog X-ray', ha='center', va='bottom', fontsize=10)

    # Major Component: Proposed Model (ViT Base)
    model_rect = plt.Rectangle((3.5, 0.8), 9.5, 5.5, fc='#e0f2f7', ec='black', lw=1.5)
    ax.add_patch(model_rect)
    ax.text(8.25, 6.5, 'Proposed Model (ViT Base)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Backbone
    backbone_rect = plt.Rectangle((4, 4.5), 2.5, 0.8, fc='#a7d9ed', ec='blue', lw=1)
    ax.add_patch(backbone_rect)
    ax.text(5.25, 4.9, 'ViT Backbone', ha='center', va='center', fontsize=9)

    # Cross-Attention
    ca_rect = plt.Rectangle((7, 3.5), 2.5, 0.8, fc='#73c2fb', ec='green', lw=1)
    ax.add_patch(ca_rect)
    ax.text(8.25, 3.9, 'Cross-Attention', ha='center', va='center', fontsize=9)

    # Keypoint Head
    kp_head_rect = plt.Rectangle((4, 1.5), 2.5, 0.8, fc='#42a5f5', ec='purple', lw=1)
    ax.add_patch(kp_head_rect)
    ax.text(5.25, 1.9, 'Keypoint Head (HRNet)', ha='center', va='center', fontsize=9)

    # Classification Head
    cls_head_rect = plt.Rectangle((10, 4.5), 2.5, 0.8, fc='#ffcc80', ec='orange', lw=1)
    ax.add_patch(cls_head_rect)
    ax.text(11.25, 4.9, 'Classification Head', ha='center', va='center', fontsize=9)

    # VHS Head
    vhs_head_rect = plt.Rectangle((10, 1.5), 2.5, 0.8, fc='#ffab91', ec='red', lw=1)
    ax.add_patch(vhs_head_rect)
    ax.text(11.25, 1.9, 'VHS Head', ha='center', va='center', fontsize=9)

    # Arrows
    ax.annotate('', xy=(3.5, 3.5), xytext=(0.2 + img_width, 3.5), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
    ax.annotate('', xy=(7, 3.9), xytext=(6.5, 4.9), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
    ax.annotate('', xy=(5.25, 2.0), xytext=(5.25, 4.5), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
    ax.annotate('', xy=(10, 4.9), xytext=(9.5, 3.9), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
    ax.annotate('', xy=(10, 1.9), xytext=(9.5, 3.9), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    # Output Text Blocks
    kp_output_rect = plt.Rectangle((14, 1.5), 1.2, 0.8, fc='#c8e6c9', ec='black', lw=1.5)
    ax.add_patch(kp_output_rect)
    ax.text(14.6, 1.9, 'Keypoint\nHeatmaps', ha='center', va='center', fontsize=9)

    cls_output_rect = plt.Rectangle((14, 4.5), 1.2, 0.8, fc='#c8e6c9', ec='black', lw=1.5)
    ax.add_patch(cls_output_rect)
    ax.text(14.6, 4.9, 'Class\nLogits', ha='center', va='center', fontsize=9)

    vhs_output_rect = plt.Rectangle((14, 3), 1.2, 0.8, fc='#c8e6c9', ec='black', lw=1.5)
    ax.add_patch(vhs_output_rect)
    ax.text(14.6, 3.4, 'VHS\nPrediction', ha='center', va='center', fontsize=9)

    final_output_rect = plt.Rectangle((14, 0), 1.2, 1.2, fc='#c8e6c9', ec='black', lw=1.5)
    ax.add_patch(final_output_rect)
    ax.text(14.6, 0.6, 'Diagnosis:\nSmall,\nNormal,\nLarge', ha='center', va='center', fontsize=9)

    ax.annotate('', xy=(14, 1.9), xytext=(12.5, 1.9), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
    ax.annotate('', xy=(14, 4.9), xytext=(12.5, 4.9), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
    ax.annotate('', xy=(14, 3.4), xytext=(12.5, 3.4), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Saved architecture figure to {out_path}")

out = '/home/chenhui/fabscore/agents4sci_rej/submission_126/fabscore_claude/workspace/model_architecture_claim55.png'
create_model_architecture_figure(out)

# Print summary of components
print("Architecture figure components:")
print("  - Title: 'Proposed Model (ViT Base)'")
print("  - Input: Dog X-ray image")
print("  - ViT Backbone")
print("  - Cross-Attention")
print("  - Keypoint Head (HRNet)")
print("  - Classification Head")
print("  - VHS Head")
print("  - Outputs: Keypoint Heatmaps, Class Logits, VHS Prediction, Diagnosis")
