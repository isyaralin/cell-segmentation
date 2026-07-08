# Isyar Ilsu Alin
### Important Note
I started to implement this project on my private university gitLab account on January 2026 and worked on it for 3 months.
After approval of my supervisor I pushed the existing progress into my personal GitHub account.
The progress is set to be 8 to 9 months and will be extended to a more detailed project in the future.

# Semi-Automatic Cell Segmentation in Time Lapse Microscopy Using First-Frame Initialization 
The aim of this project is to generate synthetic microscopy data using TRAgen, and build a pipeline
that uses the ground truth of the first frame to automatically segment all subsequent frames.
The pipeline uses Cellpose as the segmentation model and evaluates results against TRAgen ground truth masks.


## Structure (Current)

pipeline/
	pipeline.py  - baseline segmentation on all frames
	evaluate.py - evaluation of metrics (Io, Dice, F1) 
	visualize.py - visualization of results 

first_frame/
	first_frame_init.py - segmentation guided by first frame annotation 
	# will add more files later on 

## Usage
###  Baseline 
1. Run Cellpose segmentation:
	```python3 pipeline/pipeline.py```

2. Evaluate:
	```python3 pipeline/visualize.py```

3. Generate visualizations:
	```python3 pipeline/visualize.py```
### First-Frame Initialization 
	```python3 first_frame/first_frame_init.py```

## Baseline Results (83 frames, Cellpose cyto3)) 
- Mean IoU: 0.8520
- Mean Dice: 0.9200
- Mean F1: 0.9200

## Next Step 
- Evaluate first-frame initialization results and compare with baseline
- Add results plots (IoU over time)
- Generate larger dataset for final evaluation 

 



Updated
