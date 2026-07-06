import numpy as np
import tifffile
import os
import glob
import csv

GT_DIR = os.path.expanduser("~/TRAgen/dataset1")
PRED_DIR = os.path.expanduser("~/TRAgen/output_masks")
CSV_OUT = os.path.expanduser("~/TRAgen/results.csv")

gt_files = sorted(glob.glob(os.path.join(GT_DIR, "mask?????.tif")))

pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "pred_mask?????.tif")))


if len(gt_files) == 0:
	raise FileNotFoundError(f"No ground truth files found in {GT_DIR}")

if len(pred_files) == 0:
	raise FileNotFoundError(f"No prediction files found in {PRED_DIR}")

if len(gt_files) != len(pred_files): 
	raise ValueError(f"Mismatch: {len(gt_files)} GT masks but " 
			 f"{len(pred_files)} predicted masks.") 
results = []

for gt_path, pred_path in zip(gt_files, pred_files):

    gt = tifffile.imread(gt_path)
    pred = tifffile.imread(pred_path)

    gt_cells = len(np.unique(gt)) - 1
    pred_cells = len(np.unique(pred)) - 1

    count_error = abs(
        gt_cells - pred_cells
    )

    gt_bin = gt > 0
    pred_bin = pred > 0

    intersection = np.logical_and(
        gt_bin,
        pred_bin
    ).sum()

    union = np.logical_or(
        gt_bin,
        pred_bin
    ).sum()

    gt_pixels = gt_bin.sum()
    pred_pixels = pred_bin.sum()

    iou = (
        intersection / union
        if union > 0
        else 0.0
    )

    dice = (
        2 * intersection /
        (gt_pixels + pred_pixels)
        if (gt_pixels + pred_pixels) > 0
        else 0.0
    )

    precision = (
        intersection / pred_pixels
        if pred_pixels > 0
        else 0.0
    )

    recall = (
        intersection / gt_pixels
        if gt_pixels > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    frame = (
        os.path.basename(gt_path)
        .replace("mask", "")
        .replace(".tif", "")
    )

    results.append({
        "frame": frame,
        "gt_cells": gt_cells,
        "pred_cells": pred_cells,
        "count_error": count_error,
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    })

    print(
        f"Frame {frame}: "
        f"GT={gt_cells:3d} "
        f"Pred={pred_cells:3d} "
        f"IoU={iou:.4f} "
        f"Dice={dice:.4f} "
        f"F1={f1:.4f}"
    )

ious = [r["iou"] for r in results]
dices = [r["dice"] for r in results]
f1s = [r["f1"] for r in results]

print("\n" + "=" * 60)
print(f"Mean IoU:  {np.mean(ious):.4f}")
print(f"Mean Dice: {np.mean(dices):.4f}")
print(f"Mean F1:   {np.mean(f1s):.4f}")
print(f"Min IoU:   {np.min(ious):.4f}")
print(f"Max IoU:   {np.max(ious):.4f}")
print("=" * 60)

with open(CSV_OUT, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "frame",
            "gt_cells",
            "pred_cells",
            "count_error",
            "iou",
            "dice",
            "precision",
            "recall",
            "f1"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {CSV_OUT}")
