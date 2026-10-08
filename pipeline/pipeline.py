import os
import glob
import time 
import argparse 
import numpy as np
import tifffile
from cellpose import models

# Paths for the code
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset1")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_masks")

os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser(
	description = "Run Cellpose segmentation on all or selected frames")

parser.add_argument(
	"--frames",
	nargs="*",
	help="Optional frmae numbers to process, for example --frames 20 35 72. "
	     "If omitted, all frames are processed.")

args = parser.parse_args()



# Check if the dataset exists 
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATA_DIR}\n\n"
        "Place the TRAgen dataset inside the repository root."
    )

# Load images
image_files = sorted(glob.glob(os.path.join(DATA_DIR, "img*.tif")))

if len(image_files) == 0:
    raise FileNotFoundError(f"No image files found inside:\n{DATA_DIR}")


if args.frames:
	selected_frames = {
		frame.zfill(5)
		for frame in args.frames
		if frame.isdigit()
	}

	image_files = [
		image_path
		for image_path in image_files
		if os.parh.basename(image_path)
		.replace("img", "")
		.replace(".tif", "")
		in selected_frames
	]

	if len(image_files) == 0:
		raise FileNotFoundError(
			"None of the requested frames were found in the dataset"
		)


print("=" * 60)
print("TRAgen Cell Segmentation Baseline")
print("=" * 60)
print(f"Dataset directory : {DATA_DIR}")
print(f"Output directory : {OUTPUT_DIR}")
print(f"Frames found : {len(image_files)}")

if args.frames:
	print(f"Selected frames: {', '.join(args.frames)}")
else:
print("Selected frames: ALL")

# Load first-frame annotation
# I will implement this as the  next step of the project

first_mask_path = os.path.join(DATA_DIR, "mask00000.tif")

if not os.path.exists(first_mask_path):
    raise FileNotFoundError("First ground-truth mask (mask00000.tif) not found.")

first_mask = tifffile.imread(first_mask_path)

print(f"First-frame annotation loaded ({len(np.unique(first_mask)) - 1} annotated cells).")

print(
    "\nNOTE: The first annotation is not  used by the baseline "
    "Cellpose segmentation. It is loaded because it will be "
    "used later for first-frame initialization and cell "
    "identity propagation."
)

# Load Cellpose
print("\nLoading Cellpose model...")
model = models.CellposeModel(gpu=True, pretrained_model="cyto3")
print("Cellpose model ready.")

# Segment all frames
print("\nStarting segmentation...\n")
total_frames = len(image_files)

segmentation_times = []

for index, image_path in enumerate(image_files, start=1):
    try:
	# Get actual frame filename
	frame_filename = os.path.basename(image_path)
	
	# Get frame number
	frame_number = (
		frame_filename
		.replace("img", "")
		.replace(".tif", "")
	) 

	# Load image 
        image = tifffile.imread(image_path).astype(np.float32)
        
	#Normalize image 
	image = (image - image.min()) / (image.max() - image.min() + 1e-8)
	
	# Measure Cellpose segmentation time only
	start_time = time.perf_counter()

        masks, flows, styles = model.eval(
            image,
            diameter=None,
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )
	
	elapsed_time = time.perf_counter() - start_time
	segmentation_times.append(elapsed_time)

	output_path = os.path.join(
		OUTPUT_DIR,
		f"pred_mask{frame_number}.tif"
	)


        tifffile.imwrite(output_path, masks.astype(np.uint16))

        detected_cells = len(np.unique(masks)) - 1

        print(
            f"[{index:2d}/{total_frames}] "
            f"Frame {frame_filename} | "
            f"{detected_cells:3d} cells detected | "
	    f"{elapsed_time:.3f} s | "
            f"Saved -> {os.path.basename(output_path)}"
        )

    except Exception as e:
        print(f"Error while processing {os.path.basename(image_path)}")
        print(e)


print("\n" + "=" * 60)
print("Segmentation completed successfully.")
print(f"Predicted masks saved to:\n{OUTPUT_DIR}")

if segmentation_times:
	total_segmentation_time = sum(segmentation_times)
	
	average_segmentation_time = (total_segmentation_time / len(segmentation_times))
	
	print("\n========= Runtime ==========")
	
	print(
	    f"Images segmented: "
	    f"{len(segmentation_times)}"
	)
	
	print(
	    f"Total segmentation time: "
	    f"{total_segmentation_time:.3f} seconds"
	)

	print(
	   f"Average time per image: "
	   f"{average_segmentation_time:.3f} seconds"
	) 
	
	print(
	   f"Minimum time per image: "
	   f"{min(segmentation_times):.3f} seconds"
	) 
	
	print(
	   f"Maximum time per image: "
	   f"{max(segmentation_times):.3f} seconds"

print("=" * 60)

