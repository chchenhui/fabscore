import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib.image as mpimg
import os

plt.style.use('seaborn-paper')
sns.set_palette("husl")

# Load the results
with open(r'Results/results.json', 'r') as f:
    results = json.load(f)

# Create a figure with two subplots in one row for loss and accuracy curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot training and validation loss
ax1.plot(results['history']['train_loss'], label='Train Loss')
ax1.plot(results['history']['val_loss'], label='Validation Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot training and validation accuracy
ax2.plot(results['history']['train_accuracy'], label='Train Accuracy')
ax2.plot(results['history']['val_accuracy'], label='Validation Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig(r'paper/Figures/curves.png', dpi=300)
plt.close()


# --- New Figure: Proposed Model Architecture ---
def create_model_architecture_figure():
    fig, ax = plt.subplots(1, 1, figsize=(16, 8)) # Wider overall figure size
    ax.set_aspect('equal')
    ax.set_xlim(0, 16) 
    ax.set_ylim(0, 8)  
    ax.axis('off')

    # Input Image
    img = mpimg.imread(r'Data/Train/Images/2_10.8_0.png') 
    # Image dimensions: 2000x1600 (width x height) -> aspect ratio 1.25 (5:4)
    # extent=[left, right, bottom, top]
    # To maintain aspect ratio, (right-left) / (top-bottom) should be 1.25
    img_width = 2.0
    img_height = img_width / 1.25 # 1.6
    ax.imshow(img, extent=[0.2, 0.2 + img_width, 3.5 - img_height/2, 3.5 + img_height/2]) 
    ax.text(0.2 + img_width/2, 3.5 + img_height/2 + 0.2, 'Input: Dog X-ray', ha='center', va='bottom', fontsize=10)

    # Major Component: Proposed Model (ViT Base) - Wider size and centered
    model_rect = plt.Rectangle((3.5, 0.8), 9.5, 5.5, fc='#e0f2f7', ec='black', lw=1.5) # Wider
    ax.add_patch(model_rect)
    ax.text(8.25, 6.5, 'Proposed Model (ViT Base)', ha='center', va='bottom', fontsize=10, fontweight='bold') # Larger font

    # Subcomponents within Proposed Model
    # Backbone
    backbone_rect = plt.Rectangle((4, 4.5), 2.5, 0.8, fc='#a7d9ed', ec='blue', lw=1) # Adjusted size and position
    ax.add_patch(backbone_rect)
    ax.text(5.25, 4.9, 'ViT Backbone', ha='center', va='center', fontsize=9) # Larger font

    # Cross-Attention
    ca_rect = plt.Rectangle((7, 3.5), 2.5, 0.8, fc='#73c2fb', ec='green', lw=1) # Adjusted size and position
    ax.add_patch(ca_rect)
    ax.text(8.25, 3.9, 'Cross-Attention', ha='center', va='center', fontsize=9) # Larger font

    # Keypoint Head
    kp_head_rect = plt.Rectangle((4, 1.5), 2.5, 0.8, fc='#42a5f5', ec='purple', lw=1) # Adjusted size and position
    ax.add_patch(kp_head_rect)
    ax.text(5.25, 1.9, 'Keypoint Head (HRNet)', ha='center', va='center', fontsize=9) # Larger font

    # Classification Head
    cls_head_rect = plt.Rectangle((10, 4.5), 2.5, 0.8, fc='#ffcc80', ec='orange', lw=1) # Adjusted position and size
    ax.add_patch(cls_head_rect)
    ax.text(11.25, 4.9, 'Classification Head', ha='center', va='center', fontsize=9) # Larger font

    # VHS Head
    vhs_head_rect = plt.Rectangle((10, 1.5), 2.5, 0.8, fc='#ffab91', ec='red', lw=1) # Adjusted position and size
    ax.add_patch(vhs_head_rect)
    ax.text(11.25, 1.9, 'VHS Head', ha='center', va='center', fontsize=9) # Larger font

    # Arrows
    # Input to Backbone - Corrected direction
    ax.annotate('', xy=(3.5, 3.5), xytext=(0.2 + img_width, 3.5), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    # Backbone to Cross-Attention (cls_token path) - Tilted, shorter
    ax.annotate('', xy=(7, 3.9), xytext=(6.5, 4.9), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    # Backbone to Keypoint Head (patch_tokens path) - Straight, shorter
    ax.annotate('', xy=(5.25, 2.0), xytext=(5.25, 4.5), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    # Cross-Attention to Classification Head - Shorter
    ax.annotate('', xy=(10, 4.9), xytext=(9.5, 3.9), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    # Cross-Attention to VHS Head - Shorter
    ax.annotate('', xy=(10, 1.9), xytext=(9.5, 3.9), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

    # Output Text Blocks (moved to far right)
    # Keypoint Heatmaps Output
    kp_output_rect = plt.Rectangle((14, 1.5), 1.2, 0.8, fc='#c8e6c9', ec='black', lw=1.5) # Smaller height, width 1.2
    ax.add_patch(kp_output_rect)
    ax.text(14.6, 1.9, 'Keypoint\nHeatmaps', ha='center', va='center', fontsize=9) # Font size 9

    # Class Logits Output
    cls_output_rect = plt.Rectangle((14, 4.5), 1.2, 0.8, fc='#c8e6c9', ec='black', lw=1.5) # Smaller height, width 1.2
    ax.add_patch(cls_output_rect)
    ax.text(14.6, 4.9, 'Class\nLogits', ha='center', va='center', fontsize=9) # Font size 9

    # VHS Prediction Output
    vhs_output_rect = plt.Rectangle((14, 3), 1.2, 0.8, fc='#c8e6c9', ec='black', lw=1.5) # Smaller height, width 1.2
    ax.add_patch(vhs_output_rect)
    ax.text(14.6, 3.4, 'VHS\nPrediction', ha='center', va='center', fontsize=9) # Font size 9

    # Final Diagnosis Output
    final_output_rect = plt.Rectangle((14, 0), 1.2, 1.2, fc='#c8e6c9', ec='black', lw=1.5) # Height 1.2, width 1.2, changed 0.5 to 0
    ax.add_patch(final_output_rect)
    ax.text(14.6, 0.6, 'Diagnosis:\nSmall,\nNormal,\nLarge', ha='center', va='center', fontsize=9) # Font size 9, adjusted y-position

    # Arrows from Heads to their respective Outputs - Shorter
    ax.annotate('', xy=(14, 1.9), xytext=(12.5, 1.9), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8)) # KP Head to KP Output
    ax.annotate('', xy=(14, 4.9), xytext=(12.5, 4.9), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8)) # Cls Head to Cls Output
    ax.annotate('', xy=(14, 3.4), xytext=(12.5, 3.4), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8)) # VHS Head to VHS Output

    plt.savefig(r'paper/Figures/model_architecture.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()

# Call the new function to generate the figure
create_model_architecture_figure()

print("Figures created successfully in the paper/Figures folder.")