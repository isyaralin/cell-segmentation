import os
import csv
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")  # for server use 
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset1")
PRED_DIR = os.path.join(BASE_DIR, "output_masks")
VIS_DIR  = os.path.join(BASE_DIR, "visualizations")
CSV_PATH = os.path.join(BASE_DIR, "pipeline", "results.csv")
os.makedirs(VIS_DIR, exist_ok=True)

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Dataset directory not found:\n{DATA_DIR}")
if not os.path.isdir(PRED_DIR):
    raise FileNotFoundError(f"Prediction directory not found:\n{PRED_DIR}\n\nRun pipeline.py first.")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found.\nRun evaluate.py first.")

# Load IoU values from CSV
iou_scores = {}
with open(CSV_PATH, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        iou_scores[row["frame"]] = float(row["iou"])

# Frames to visualize
# Produce small amount (begining, during and the end)
frames_to_show = [0, 20, 40, 60, 82]

print("=" * 60)
print("Generating Visualizations")
print("=" * 60)

# Generate figures
for frame in frames_to_show:
    frame_name = f"{frame:05d}"
    image_path = os.path.join(DATA_DIR, f"img{frame_name}.tif")
    gt_path  = os.path.join(DATA_DIR, f"mask{frame_name}.tif")
    pred_path = os.path.join(PRED_DIR, f"pred_mask{frame_name}.tif")

    if not os.path.exists(pred_path):
        print(f"Prediction for frame {frame_name} not found, skipping.")
        continue

    image = tifffile.imread(image_path).astype(np.float32)
    gt = tifffile.imread(gt_path)
    pred = tifffile.imread(pred_path)

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    gt_cells = len(np.unique(gt)) - 1
    pred_cells = len(np.unique(pred)) - 1
    iou = iou_scores.get(frame_name, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Frame {frame_name} | IoU = {iou:.4f}",
        fontsize=15,
        fontweight="bold"
    )

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="nipy_spectral")
    axes[1].set_title(f"Ground Truth\n{gt_cells} annotated cells")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="nipy_spectral")
    axes[2].set_title(f"Cellpose Prediction\n{pred_cells} detected cells")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(VIS_DIR, f"vis_frame{frame_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization: vis_frame{frame_name}.png")

print("\n" + "=" * 60)
print("Visualization completed successfully.")
print(f"Images saved to:\n{VIS_DIR}")
print("=" * 60)
