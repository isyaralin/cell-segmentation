import numpy as np
import tifffile
import os
import glob
from cellpose import models

# Configurations for the files 
data_dir = "/home/isyalin/TRAgen/dataset1/"
output_dir = "/home/isyalin/TRAgen/output_masks/"
os.makedirs(output_dir, exist_ok=True)

# Load the frames in order to process 
image_files = sorted(glob.glob(os.path.join(data_dir, "img?????.tif")))
print(f"Found {len(image_files)} frames")

# Load the first frame ground truth mask as the first annotation 
first_mask_path = os.path.join(data_dir, "mask00000.tif")
first_mask = tifffile.imread(first_mask_path)
print(f"First frame mask loaded: {first_mask.shape}, {len(np.unique(first_mask))-1} cells")

# Cellpose model initialization 
print("\nLoading Cellpose model")
model = models.CellposeModel(gpu=True, pretrained_model="cyto3")
print("Model ready!")

# run segmentation on all frames 
print("\nSegmenting all frames")
for i, img_path in enumerate(image_files):

    # Load image and normalize to float32
    image = tifffile.imread(img_path).astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Run Cellpose
    masks, flows, styles = model.eval(
        image,
        diameter=None,   # estimate cell diameter automatically 
        channels=[0, 0], 
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )

    # Save output mask
    frame_num = os.path.basename(img_path).replace("img", "").replace(".tif", "")
    out_path = os.path.join(output_dir, f"pred_mask{frame_num}.tif")
    tifffile.imwrite(out_path, masks.astype(np.uint16))

    print(f"  Frame {i:3d}: {len(np.unique(masks))-1} cells detected → saved {os.path.basename(out_path)}")

print(f"\nDone! All masks saved to {output_dir}")
