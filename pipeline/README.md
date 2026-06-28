# Cell Segmentation Pipeline

Semi-automatic cell segmentation in time-lapse microscopy using first-frame initialization.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install cellpose tifffile matplotlib torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage
```bash
python3 pipeline.py
python3 evaluate.py
python3 visualize.py
```

## Baseline Results (Cellpose cyto3, no first-frame initialization)
- Frames evaluated: 83
- Mean IoU:  0.8520
- Mean Dice: 0.9200
