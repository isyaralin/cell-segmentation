import numpy as np
import tifffile
import os
import glob
import csv

GT_DIR   = "/home/isyalin/TRAgen/dataset1/"
PRED_DIR = "/home/isyalin/TRAgen/output_masks/"
CSV_OUT  = "/home/isyalin/TRAgen/results.csv"

gt_files   = sorted(glob.glob(os.path.join(GT_DIR,   "mask?????.tif")))
pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "pred_mask?????.tif")))

print(f"Evaluating {len(gt_files)} frames...\n")

results = []
for gt_path, pred_path in zip(gt_files, pred_files):
    gt   = tifffile.imread(gt_path)
    pred = tifffile.imread(pred_path)

    gt_cells   = len(np.unique(gt))   - 1
    pred_cells = len(np.unique(pred)) - 1

    gt_bin   = (gt   > 0)
    pred_bin = (pred > 0)

    intersection = np.logical_and(gt_bin, pred_bin).sum()
    union        = np.logical_or(gt_bin,  pred_bin).sum()
    iou  = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / (gt_bin.sum() + pred_bin.sum()) if (gt_bin.sum() + pred_bin.sum()) > 0 else 0.0

    frame = os.path.basename(gt_path).replace("mask","").replace(".tif","")
    results.append({"frame": frame, "gt_cells": gt_cells, "pred_cells": pred_cells, "iou": round(iou,4), "dice": round(dice,4)})
    print(f"  Frame {frame}: GT={gt_cells:3d}  Pred={pred_cells:3d}  IoU={iou:.4f}  Dice={dice:.4f}")

ious  = [r["iou"]  for r in results]
dices = [r["dice"] for r in results]
print(f"\n{'='*55}")
print(f"  Mean IoU:  {np.mean(ious):.4f}")
print(f"  Mean Dice: {np.mean(dices):.4f}")
print(f"  Min IoU:   {np.min(ious):.4f}  (frame {results[np.argmin(ious)]['frame']})")
print(f"  Max IoU:   {np.max(ious):.4f}  (frame {results[np.argmax(ious)]['frame']})")
print(f"{'='*55}")

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["frame","gt_cells","pred_cells","iou","dice"])
    writer.writeheader()
    writer.writerows(results)
print(f"\nResults saved to {CSV_OUT}")
