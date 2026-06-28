import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DATA_DIR = "/home/isyalin/TRAgen/dataset1/"
PRED_DIR = "/home/isyalin/TRAgen/output_masks/"
VIS_DIR  = "/home/isyalin/TRAgen/visualizations/"
os.makedirs(VIS_DIR, exist_ok=True)

# Visualize frames 0, 20, 40, 60, 82
frames_to_show = [0, 20, 40, 60, 82]

for frame_num in frames_to_show:
    img  = tifffile.imread(os.path.join(DATA_DIR, f"img{frame_num:05d}.tif")).astype(np.float32)
    gt   = tifffile.imread(os.path.join(DATA_DIR, f"mask{frame_num:05d}.tif"))
    pred = tifffile.imread(os.path.join(PRED_DIR, f"pred_mask{frame_num:05d}.tif"))

    # For display normalize the images 
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Frame {frame_num:05d}", fontsize=14, fontweight='bold')

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(gt, cmap='nipy_spectral')
    axes[1].set_title(f"Ground Truth ({len(np.unique(gt))-1} cells)")
    axes[1].axis('off')

    axes[2].imshow(pred, cmap='nipy_spectral')
    axes[2].set_title(f"Cellpose Prediction ({len(np.unique(pred))-1} cells)")
    axes[2].axis('off')

    plt.tight_layout()
    out_path = os.path.join(VIS_DIR, f"vis_frame{frame_num:05d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

print("\nDone! Download visualizations to see results.")
