# existing evaluate.py evaluates pixels 
# is each pixel classified as cell or background?
# This is handled by converting to binary, 0 for background 1 for cell 

# evaluate_instances.py evaluates individual cells
# does each predicted cell correspond to the correct ground-truth cell?
# how accurately does it overlap?

# PQ considers 2 things
# Recognition Quality (RQ): Did the model identify the individual cells correctly?
# Segmentation Quality (SQ): How accurately did the predicted masks overlap with their corresponding ground-truth masks?

# Then calculates PQ = SQ x RQ

import os
import glob
import csv
import argparse
import numpy as np
import tifffile

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.path.join(BASE_DIR, "dataset1")
PRED_DIR = os.path.join(BASE_DIR, "output_masks")
CSV_OUT = os.path.join(BASE_DIR, "pipeline", "results_pq.csv")

# Check whether the folder exist.  
if not os.path.isdir(GT_DIR):
    raise FileNotFoundError(f"Ground-truth directory not found:\n{GT_DIR}")

if not os.path.isdir(PRED_DIR):
    raise FileNotFoundError(f"Prediction directory not found:\n{PRED_DIR}")

# Manually check selected frames. 
parser = argparse.ArgumentParser(
    description="Evaluate instance segmentation using Panoptic Quality"
)

parser.add_argument(
    "--frames",
    nargs="+",
    help="Optional frame numbers, for example --frames 0 20 72"
)

args = parser.parse_args()

# Find the TIFF files 
gt_files = sorted(glob.glob(os.path.join(GT_DIR, "mask*.tif"))) # for files with tif extension 
pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "pred_mask*.tif"))) # same thing for predictions

def get_frame_number(path, prefix):
    return os.path.basename(path).replace(prefix, "").replace(".tif", "")

# Create dictionaries that store ket as the given mask names and their corresponding file path 
# For mask and the pred_mask
gt_by_frame = {
    get_frame_number(path, "mask"): path
    for path in gt_files
}

pred_by_frame = {
    get_frame_number(path, "pred_mask"): path
    for path in pred_files
}

# Check whether the files are were 
if not gt_by_frame:
    raise RuntimeError("No ground-truth masks found.")

if not pred_by_frame:
    raise RuntimeError("No predicted masks found.")

matching_frames = sorted(set(gt_by_frame) & set(pred_by_frame))

if args.frames:
    requested_frames = {str(int(frame)).zfill(5) for frame in args.frames}
    matching_frames = [
        frame for frame in matching_frames
        if frame in requested_frames
    ]

if not matching_frames:
    raise RuntimeError("No matching ground-truth and prediction frames found.")

