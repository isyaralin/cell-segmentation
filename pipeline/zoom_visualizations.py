import os
import csv
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPOSITORY_ROOT, "dataset1")
PREDICTIONS_DIR = os.path.join(REPOSITORY_ROOT, "output_masks")
VISUALIZATIONS_DIR = os.path.join(REPOSITORY_ROOT, "visualizations")
EVALUATION_CSV = os.path.join(REPOSITORY_ROOT, "pipeline", "results.csv")
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(f"Dataset folder not found:\n{DATASET_DIR}")
if not os.path.isdir(PREDICTIONS_DIR):
    raise FileNotFoundError(f"Predictions folder not found:\n{PREDICTIONS_DIR}\nRun pipeline.py first.")
if not os.path.exists(EVALUATION_CSV):
    raise FileNotFoundError(f"Evaluation results not found:\n{EVALUATION_CSV}\nRun evaluate.py first.")

evaluation_results = []
with open(EVALUATION_CSV, newline="") as csv_file:
    for row in csv.DictReader(csv_file):
        row["iou"] = float(row["iou"])
        evaluation_results.append(row)

if len(evaluation_results) == 0:
    raise RuntimeError("results.csv is empty. Run evaluate.py first.")

worst_frame_result = min(evaluation_results, key=lambda row: row["iou"])
worst_frame_name = worst_frame_result["frame"]
worst_frame_iou = worst_frame_result["iou"]

print("\n========== Worst Frame ==========")
print(f"Frame: {worst_frame_name}")
print(f"IoU: {worst_frame_iou:.4f}")

microscopy_image = tifffile.imread(os.path.join(DATASET_DIR, f"img{worst_frame_name}.tif")).astype(np.float32)
ground_truth_mask = tifffile.imread(os.path.join(DATASET_DIR, f"mask{worst_frame_name}.tif"))
predicted_mask = tifffile.imread(os.path.join(PREDICTIONS_DIR, f"pred_mask{worst_frame_name}.tif"))

microscopy_image = (microscopy_image - microscopy_image.min()) / (microscopy_image.max() - microscopy_image.min() + 1e-8)

def calculate_cell_iou(gt_cell, pred_mask):
    overlapping_labels = pred_mask[gt_cell]
    overlapping_labels = overlapping_labels[overlapping_labels != 0]

    if len(overlapping_labels) == 0:
        return 0.0

    matched_prediction_label = np.bincount(overlapping_labels).argmax()
    matched_prediction_cell  = pred_mask == matched_prediction_label

    intersection = np.logical_and(gt_cell, matched_prediction_cell).sum()
    union = np.logical_or(gt_cell,  matched_prediction_cell).sum()

    if union == 0:
        return 0.0

    return intersection / union

all_cell_labels = np.unique(ground_truth_mask)
all_cell_labels = all_cell_labels[all_cell_labels != 0]
worst_cell_label = None
worst_cell_iou = 1.0

for cell_label in all_cell_labels:
    gt_cell  = ground_truth_mask == cell_label
    cell_iou = calculate_cell_iou(gt_cell, predicted_mask)
    if cell_iou < worst_cell_iou:
        worst_cell_iou   = cell_iou
        worst_cell_label = cell_label

print(f"\n========== Worst Cell ==========")
print(f"Cell label: {worst_cell_label}")
print(f"Cell IoU: {worst_cell_iou:.4f}")

worst_cell = ground_truth_mask == worst_cell_label
row_has_cell = np.any(worst_cell, axis=1)
col_has_cell = np.any(worst_cell, axis=0)
row_start, row_end = np.where(row_has_cell)[0][[0, -1]]
col_start, col_end = np.where(col_has_cell)[0][[0, -1]]

ZOOM_PADDING = 60
row_start = max(0, row_start - ZOOM_PADDING)
row_end = min(microscopy_image.shape[0], row_end + ZOOM_PADDING)
col_start = max(0, col_start - ZOOM_PADDING)
col_end  = min(microscopy_image.shape[1], col_end + ZOOM_PADDING)

overlapping_labels = predicted_mask[worst_cell]
overlapping_labels = overlapping_labels[overlapping_labels != 0]

if len(overlapping_labels) == 0:
    matched_prediction_cell = np.zeros_like(predicted_mask, dtype=bool)
else:
    matched_prediction_label = np.bincount(overlapping_labels).argmax()
    matched_prediction_cell  = predicted_mask == matched_prediction_label

zoomed_microscopy = microscopy_image [row_start:row_end, col_start:col_end]
zoomed_ground_truth = worst_cell [row_start:row_end, col_start:col_end]
zoomed_prediction = matched_prediction_cell[row_start:row_end, col_start:col_end]
full_difference_map = np.logical_xor(worst_cell, matched_prediction_cell)
zoomed_difference_map = full_difference_map [row_start:row_end, col_start:col_end]

zoom_box = dict(linewidth=2, edgecolor="red", facecolor="none")

figure = plt.figure(figsize=(20, 10))
figure.suptitle(
    f"Worst Cell Analysis\n"
    f"Frame {worst_frame_name} | Frame IoU = {worst_frame_iou:.4f} | Worst Cell IoU = {worst_cell_iou:.4f}",
    fontsize=16, fontweight="bold"
)

full_image_panel = figure.add_subplot(2, 4, 1)
full_image_panel.imshow(microscopy_image, cmap="gray")
full_image_panel.add_patch(Rectangle((col_start, row_start), col_end-col_start, row_end-row_start, **zoom_box))
full_image_panel.set_title("Original Image\n(red box = worst cell region)")
full_image_panel.axis("off")

full_gt_panel = figure.add_subplot(2, 4, 2)
full_gt_panel.imshow(ground_truth_mask, cmap="nipy_spectral")
full_gt_panel.add_patch(Rectangle((col_start, row_start), col_end-col_start, row_end-row_start, **zoom_box))
full_gt_panel.set_title(f"Ground Truth (Manual)\n({len(all_cell_labels)} cells)")
full_gt_panel.axis("off")

full_pred_panel = figure.add_subplot(2, 4, 3)
full_pred_panel.imshow(predicted_mask, cmap="nipy_spectral")
full_pred_panel.add_patch(Rectangle((col_start, row_start), col_end-col_start, row_end-row_start, **zoom_box))
full_pred_panel.set_title(f"Cellpose Prediction\n({len(np.unique(predicted_mask))-1} cells)")
full_pred_panel.axis("off")

full_diff_panel = figure.add_subplot(2, 4, 4)
full_diff_panel.imshow(full_difference_map, cmap="gray")
full_diff_panel.set_title("Selected Cell Difference\n(white = disagreement)")
full_diff_panel.axis("off")

zoomed_image_panel = figure.add_subplot(2, 4, 5)
zoomed_image_panel.imshow(zoomed_microscopy, cmap="gray")
zoomed_image_panel.set_title("Zoomed Original")
zoomed_image_panel.axis("off")

zoomed_gt_panel = figure.add_subplot(2, 4, 6)
zoomed_gt_panel.imshow(zoomed_ground_truth, cmap="gray")
zoomed_gt_panel.set_title("Worst Cell\n(Ground Truth)")
zoomed_gt_panel.axis("off")

zoomed_pred_panel = figure.add_subplot(2, 4, 7)
zoomed_pred_panel.imshow(zoomed_prediction, cmap="gray")
zoomed_pred_panel.set_title("Matched Cell\n(Cellpose)")
zoomed_pred_panel.axis("off")

zoomed_diff_panel = figure.add_subplot(2, 4, 8)
zoomed_diff_panel.imshow(zoomed_difference_map, cmap="gray")
zoomed_diff_panel.set_title("Zoomed Cell Difference\n(white = disagreement)")
zoomed_diff_panel.axis("off")

plt.tight_layout()
output_path = os.path.join(VISUALIZATIONS_DIR, f"zoom_frame{worst_frame_name}.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"\nWorst cell IoU: {worst_cell_iou:.4f}")
print(f"Saved: {output_path}")

