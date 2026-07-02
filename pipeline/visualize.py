import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.expanduser("~/TRAgen/dataset1")
PRED_DIR = os.path.expanduser("~/TRAgen/output_masks")
VIS_DIR = os.path.expanduser("~/TRAgen/visualizations")

os.makedirs(VIS_DIR, exist_ok=True)

frames_to_show = [0, 20, 40, 60, 82]

for frame_num in frames_to_show:

    img = tifffile.imread(
        os.path.join(
            DATA_DIR,
            f"img{frame_num:05d}.tif"
        )
    ).astype(np.float32)

    gt = tifffile.imread(
        os.path.join(
            DATA_DIR,
            f"mask{frame_num:05d}.tif"
        )
    )

    pred = tifffile.imread(
        os.path.join(
            PRED_DIR,
            f"pred_mask{frame_num:05d}.tif"
        )
    )

    img = (
        img - img.min()
    ) / (
        img.max() - img.min() + 1e-8
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    fig.suptitle(
        f"Frame {frame_num:05d}",
        fontsize=14,
        fontweight="bold"
    )

    axes[0].imshow(
        img,
        cmap="gray"
    )
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(
        img,
        cmap="gray"
    )
    axes[1].imshow(
        gt,
        cmap="nipy_spectral",
        alpha=0.5
    )
    axes[1].set_title(
        f"Ground Truth "
        f"({len(np.unique(gt))-1} cells)"
    )
    axes[1].axis("off")

    axes[2].imshow(
        img,
        cmap="gray"
    )
    axes[2].imshow(
        pred,
        cmap="nipy_spectral",
        alpha=0.5
    )
    axes[2].set_title(
        f"Prediction "
        f"({len(np.unique(pred))-1} cells)"
    )
    axes[2].axis("off")

    plt.tight_layout()

    out_path = os.path.join(
        VIS_DIR,
        f"vis_frame{frame_num:05d}.png"
    )

    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved {os.path.basename(out_path)}"
    )

print("\nDone! Visualizations saved.")
