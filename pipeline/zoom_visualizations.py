import os
import csv
import numpy as np
import tifffile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(REPOSITORY_ROOT, "dataset1")

PREDICTIONS_DIR = os.path.join(REPOSITORY_ROOT, "output_masks")

VISUALIZATIONS_DIR = os.path.join(REPOSITORY_ROOT, "visualizations")

EVALUATION_CSV = os.path.join(REPOSITORY_ROOT, "pipeline", "results.csv")

os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_DIR}"
    )

if not os.path.isdir(PREDICTIONS_DIR):
    raise FileNotFoundError(
        f"Predictions folder not found:\n"
        f"{PREDICTIONS_DIR}\n"
        f"Run pipeline.py first."
    )

if not os.path.exists(EVALUATION_CSV):
    raise FileNotFoundError(
        f"Evaluation results not found:\n"
        f"{EVALUATION_CSV}\n"
        f"Run evaluate.py first."
    )

evaluation_results = []

with open(EVALUATION_CSV, newline="") as csv_file:

    reader = csv.DictReader(csv_file)

    for row in reader:

        row["iou"] = float(row["iou"])
        evaluation_results.append(row)


if len(evaluation_results) == 0:

    raise RuntimeError("results.csv is empty. Run evaluate.py first.")


worst_frame_result = min(evaluation_results, key=lambda row: row["iou"])

worst_frame_name = worst_frame_result["frame"]
worst_frame_iou = worst_frame_result["iou"]


print("\n========== Center Point Visualization ==========")

print(f"Frame : {worst_frame_name}")

print(f"IoU: {worst_frame_iou:.4f}")

image_path = os.path.join(DATASET_DIR,f"img{worst_frame_name}.tif")

ground_truth_path = os.path.join(DATASET_DIR, f"mask{worst_frame_name}.tif")

prediction_path = os.path.join(PREDICTIONS_DIR, f"pred_mask{worst_frame_name}.tif")

microscopy_image = tifffile.imread(image_path).astype(np.float32)


ground_truth_mask = tifffile.imread(ground_truth_path)

predicted_mask = tifffile.imread(prediction_path)

microscopy_image = (microscopy_image - microscopy_image.min()) / (microscopy_image.max() - microscopy_image.min() + 1e-8)

def calculate_cell_centers(mask):
    """
    Calculate the center point of every labeled cell.

    Background is assumed to have label 0.

    Returns:
        list of (x, y) center coordinates.
    """

    centers = []

    labels = np.unique(mask)

    labels = labels[labels != 0]

    for label in labels:

        cell_pixels = np.where(mask == label)

        if len(cell_pixels[0]) == 0:
            continue

        center_y = np.mean(cell_pixels[0])
        center_x = np.mean(cell_pixels[1])

        centers.append((center_x, center_y))

    return centers


ground_truth_centers = calculate_cell_centers(ground_truth_mask)

prediction_centers = calculate_cell_centers(predicted_mask)

ground_truth_count = len(ground_truth_centers)

prediction_count = len(prediction_centers)

print(f"Ground Truth cells: {ground_truth_count}")

print(f"Predicted cells: {prediction_count}")

print(f"Difference: " f"{prediction_count - ground_truth_count:+d}")

figure, axis = plt.subplots(figsize=(12, 10))

axis.imshow(microscopy_image, cmap="gray")

if ground_truth_centers:

    gt_x = [center[0] for center in ground_truth_centers]

    gt_y = [center[1] for center in ground_truth_centers]

    axis.scatter(
        gt_x,
        gt_y,
        marker="s",
        s=45,
        facecolors="none",
        edgecolors="lime",
        linewidths=1.5
    )

if prediction_centers:

    pred_x = [center[0] for center in prediction_centers]

    pred_y = [center[1] for center in prediction_centers]

    axis.scatter(
        pred_x,
        pred_y,
        marker="x",
        s=45,
        color="red",
        linewidths=1.5
    )

axis.set_title(
    f"Cell Center Comparison\n"
    f"Frame {worst_frame_name} | "
    f"IoU = {worst_frame_iou:.4f}\n"
    f"Ground Truth = {ground_truth_count} cells | "
    f"Prediction = {prediction_count} cells",
    fontsize=15,
    fontweight="bold"
)


axis.axis("off")

legend_elements = [

    Line2D(
        [0],
        [0],
        marker="s",
        color="none",
        markeredgecolor="lime",
        markerfacecolor="none",
        markersize=8,
        label="Ground Truth center"
    ),

    Line2D(
        [0],
        [0],
        marker="x",
        color="red",
        markersize=8,
        linewidth=0,
        label="Prediction center"
    )
]

axis.legend(handles=legend_elements, loc="upper right", fontsize=10)

plt.tight_layout()

output_path = os.path.join(VISUALIZATIONS_DIR, f"center_frame{worst_frame_name}.png")

plt.savefig(output_path, dpi=200, bbox_inches="tight")


plt.close()

print(f"\nSaved center visualization:")

print(output_path)

