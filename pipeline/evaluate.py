import os
import glob
import csv
import numpy as np
import tifffile

# Paths for the code
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(BASE_DIR, "dataset1")
PRED_DIR = os.path.join(BASE_DIR, "output_masks")
CSV_OUT = os.path.join(BASE_DIR, "pipeline", "results.csv")

# Check if the folders exist
if not os.path.isdir(GT_DIR):
    raise FileNotFoundError(f"Ground-truth directory not found:\n{GT_DIR}")
if not os.path.isdir(PRED_DIR):
    raise FileNotFoundError(f"Prediction directory not found:\n{PRED_DIR}\n\nRun pipeline.py first.")

gt_files = sorted(glob.glob(os.path.join(GT_DIR,"mask*.tif")))
pred_files = sorted(glob.glob(os.path.join(PRED_DIR,"pred_mask*.tif")))

if len(gt_files) == 0:
    raise RuntimeError("No ground-truth masks found.")
if len(pred_files) == 0:
    raise RuntimeError("No predicted masks found.\nRun pipeline.py first.")
if len(gt_files) != len(pred_files):
    print(f"Warning: GT masks={len(gt_files)}, Predicted masks={len(pred_files)}")
    print("Only matching pairs will be evaluated.\n")

print("=" * 60)
print("Evaluating Cellpose Baseline Segmentation")
print("=" * 60)
print(f"Frames to evaluate: {min(len(gt_files), len(pred_files))}\n")

# Evaluation
results = []

for gt_path, pred_path in zip(gt_files, pred_files):
    gt = tifffile.imread(gt_path)
    pred = tifffile.imread(pred_path)

    gt_cells = len(np.unique(gt)) - 1
    pred_cells = len(np.unique(pred)) - 1
    count_error = abs(gt_cells - pred_cells)

    gt_bin = gt > 0
    pred_bin = pred > 0

    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    gt_pixels = gt_bin.sum()
    pred_pixels = pred_bin.sum()

    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / (gt_pixels + pred_pixels) if (gt_pixels + pred_pixels) > 0 else 0.0
    precision = intersection / pred_pixels if pred_pixels > 0 else 0.0
    recall = intersection / gt_pixels if gt_pixels > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    frame = os.path.basename(gt_path).replace("mask", "").replace(".tif", "")
    results.append({
        "frame": frame,
        "gt_cells": gt_cells,
        "pred_cells": pred_cells,
        "count_error": count_error,
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    })
    print(
        f"Frame {frame}: GT={gt_cells:3d} Pred={pred_cells:3d} "
        f"Error={count_error:2d} IoU={iou:.4f} "
        f"Prec={precision:.4f} Rec={recall:.4f} "
        f"Dice={dice:.4f} F1={f1:.4f}"
    )

# The statistics
ious = [r["iou"] for r in results]
dices = [r["dice"] for r in results]
precisions = [r["precision"] for r in results]
recalls = [r["recall"] for r in results]
f1s = [r["f1"] for r in results]
errors = [r["count_error"] for r in results]

print("\n" + "=" * 60)
print(f"Mean IoU: {np.mean(ious):.4f}")
print(f"Mean Dice: {np.mean(dices):.4f}")
print(f"Mean Precision: {np.mean(precisions):.4f}")
print(f"Mean Recall: {np.mean(recalls):.4f}")
print(f"Mean F1: {np.mean(f1s):.4f}")
print(f"Mean Cell Count Error: {np.mean(errors):.2f}")
print(f"Minimum IoU: {np.min(ious):.4f}")
print(f"Maximum IoU: {np.max(ious):.4f}")
print(f"Maximum Count Error: {np.max(errors)}")
print("=" * 60)

# Save as CSV
with open(CSV_OUT, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        "frame", "gt_cells", "pred_cells", "count_error",
        "iou", "dice", "precision", "recall", "f1"
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to:\n{CSV_OUT}")
