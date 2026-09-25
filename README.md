# Isyar Ilsu Alin

# Semi-Automatic Cell Segmentation in Time-Lapse Microscopy Using First-Frame Initialization

## Project Overview

The aim of this project is to develop a semi-automatic cell segmentation pipeline for time-lapse microscopy data generated using the TRAgen dataset.

The long-term goal is to use the ground-truth annotation of the first frame to initialize cell identities and then propagate these identities through subsequent frames.

The current implementation focuses on establishing and evaluating a baseline segmentation pipeline using Cellpose. The first-frame annotation is already loaded by the pipeline and prepared for the next stage of the project: first-frame initialization and cell identity propagation.

# Baseline Cell Segmentation Pipeline

The baseline pipeline uses the Cellpose `cyto3` model to segment individual cells in grayscale microscopy images.

The dataset currently contains 83 microscopy frames together with corresponding ground-truth segmentation masks.

The baseline implementation provides:

* Automated Cellpose-based cell segmentation
* Processing of all 83 microscopy frames
* Ground-truth comparison
* IoU evaluation
* Dice score evaluation
* Precision evaluation
* Recall evaluation
* F1-score evaluation
* Visualization of segmentation results
* Visualization of the worst-performing frame
* Visualization of the worst-performing cell
* Cell-level IoU analysis
* Center-point comparison between ground-truth and predicted cells
* Runtime measurements for segmentation

The baseline does not yet use the first-frame annotation to guide Cellpose segmentation.

The first-frame annotation is currently loaded and prepared for the first-frame initialization stage.

# Dataset

The project uses microscopy data generated using the TRAgen dataset.

The dataset contains:

* 83 grayscale microscopy images
* 83 corresponding ground-truth segmentation masks

The images follow the naming convention:

```text
img00000.tif
img00001.tif
...
img00082.tif
```

The corresponding masks are:

```text
mask00000.tif
mask00001.tif
...
mask00082.tif
```

# Obtaining the Dataset

The dataset is not generated automatically by the repository.

The required TRAgen dataset should be placed inside the repository root in:

```text
dataset1/
```

The directory should contain both the microscopy images and their corresponding ground-truth masks.

After placing the dataset in this directory, the pipeline can be executed normally.

## Repository Structure

```text
isyar-ilsu-alin/
│
├── README.md
├── LICENSE
├── Specification.docx
│
├── dataset1/
│   ├── img00000.tif
│   ├── img00001.tif
│   ├── ...
│   ├── img00082.tif
│   ├── mask00000.tif
│   ├── mask00001.tif
│   ├── ...
│   └── mask00082.tif
│
├── output_masks/
│   ├── pred_mask00000.tif
│   ├── pred_mask00001.tif
│   ├── ...
│   └── pred_mask00082.tif
│
├── visualizations/
│   ├── ...
│   └── generated visualization images
│
└── pipeline/
    ├── pipeline.py
    ├── evaluate.py
    ├── visualize.py
    ├── zoom_visualization.py
    ├── center_visualizations.py
    └── results.csv
```

# Requirements

The implementation uses Python and the following main libraries:

* NumPy
* tifffile
* Matplotlib
* Cellpose
* PyTorch

A Python virtual environment is recommended:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install numpy tifffile matplotlib cellpose torch
```

The Cellpose model is configured to use GPU acceleration when available.

# Usage

## 1. Run the Baseline Segmentation

From the repository root:

```bash
python3 pipeline/pipeline.py
```

This loads the Cellpose `cyto3` model and processes all microscopy frames.

The predicted masks are saved to:

```text
output_masks/
```

---

## 2. Evaluate the Segmentation

Run:

```bash
python3 pipeline/evaluate.py
```

The evaluation compares the predicted masks against the ground-truth masks.

The following metrics are calculated:

* IoU
* Dice score
* Precision
* Recall
* F1-score
* Cell count error

The results are stored in:

```text
pipeline/results.csv
```

---

## 3. Generate Segmentation Visualizations

Run:

```bash
python3 pipeline/visualize.py
```

This generates visualizations for selected segmentation results.

The visualizations are saved to:

```text
visualizations/
```

---

## 4. Generate Worst-Cell Visualization

Run:

```bash
python3 pipeline/zoom_visualization.py
```

The script automatically:

1. Finds the frame with the lowest IoU.
2. Finds the ground-truth cell with the lowest cell-level IoU in that frame.
3. Finds the predicted cell that overlaps the selected ground-truth cell the most.
4. Creates a zoomed region around the selected cell.
5. Shows the original microscopy image.
6. Shows the ground-truth segmentation.
7. Shows the matched Cellpose prediction.
8. Shows the disagreement between the two cells.

The resulting visualization is saved in:

```text
visualizations/
```

This visualization is intended to help inspect individual segmentation errors, such as inaccurate boundaries, merged cells, or incorrectly segmented cells.

---

## 5. Generate Center-Point Visualization

Run:

```bash
python3 pipeline/center_visualizations.py
```

This visualization compares the center points of the ground-truth cells and predicted cells.

The visualization uses:

* Green squares for ground-truth cell centers
* Red crosses for predicted cell centers

This makes it possible to identify errors that are difficult to understand from the segmentation masks alone.

For example, the center-point visualization can reveal:

* Missing cells
* Extra predicted cells
* Incorrect cell locations
* Merged cells
* Cases where two ground-truth cells are represented by one predicted cell
* Cases where one ground-truth cell is represented by multiple predicted cells

The resulting image is saved in:

```text
visualizations/
```

# Current Visual Analysis

Two complementary visualizations are currently implemented.

## Worst-Cell Visualization

The worst-cell visualization focuses on segmentation quality.

It compares:

* Original microscopy image
* Ground-truth cell
* Cellpose predicted cell
* Difference map

This helps identify boundary errors, merged cells, and incorrectly segmented cells.

## Center-Point Visualization

The center-point visualization focuses on object locations and cell-count differences.

It compares the spatial locations of ground-truth cell centers and predicted cell centers.

Different markers make the two sets of cell centers easier to distinguish:

* Green squares = ground-truth cell centers
* Red crosses = predicted cell centers

This is useful for understanding why the number of predicted cells may differ from the number of ground-truth cells.

# Next Steps

The next development stage is first-frame initialization and cell identity propagation.

Planned steps:

1. Use the first-frame ground-truth annotation as the initial set of cells.
2. Extract the individual cells and their center points.
3. Assign a unique identity to each initial cell.
4. Segment the next frame.
5. Calculate the center points of the detected cells.
6. Match cells between consecutive frames using spatial proximity or overlap.
7. Propagate cell identities to the next frame.
8. Handle possible cell appearance, disappearance, merging, and splitting.

