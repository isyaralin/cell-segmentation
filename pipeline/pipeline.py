import numpy as np
import tifffile
import os
import glob
from cellpose import models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "dataset1")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_masks")

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = sorted(
    glob.glob(os.path.join(DATA_DIR, "img?????.tif"))
)

print(f"Found {len(image_files)} frames.")

first_mask_path = os.path.join(
    DATA_DIR,
    "mask00000.tif"
)

first_mask = tifffile.imread(first_mask_path)

print(
    f"First frame contains "
    f"{len(np.unique(first_mask)) - 1} cells."
)

print("\nLoading Cellpose model...")

model = models.CellposeModel(
    gpu=True,
    pretrained_model="cyto3"
)

print("Model ready.")
print("\nSegmenting frames...\n")

for i, img_path in enumerate(image_files):

    image = tifffile.imread(img_path).astype(np.float32)

    image = (
        image - image.min()
    ) / (
        image.max() - image.min() + 1e-8
    )

    masks, flows, styles = model.eval(
        image,
        diameter=None,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )

    frame_num = (
        os.path.basename(img_path)
        .replace("img", "")
        .replace(".tif", "")
    )

    out_path = os.path.join(
        OUTPUT_DIR,
        f"pred_mask{frame_num}.tif"
    )

    tifffile.imwrite(
        out_path,
        masks.astype(np.uint16)
    )

    pred_cells = len(np.unique(masks)) - 1

    print(
        f"Frame {frame_num}: "
        f"{pred_cells} cells detected "
        f"-> saved {os.path.basename(out_path)}"
    )

print("\nDone! All masks saved.")
