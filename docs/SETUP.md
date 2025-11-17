# ELEC537 Assignment 3 - Setup Guide

## Project Structure

```structure
elec537-asmt3/
├── README.md
├── requirements.txt
├── docs/
│   └── SETUP.md
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config.yaml        # Configuration file
│   │   ├── utils.py           # Shared utilities
│   │   └── model_handler.py   # Model loading/inference
│   ├── task1_detection.py     # Real-time object detection
│   ├── task2.b_optimization.py # Model export & quantization
│   └── task2.c_comparison.py  # Model benchmarking
├── models/                     # Downloaded models
├── data/
│   ├── coco_classes.json
│   └── validation/            # Validation images
└── results/                    # Output directory
```

## Prerequisites

### Hardware Requirements

- Raspberry Pi 4 (recommended: 4GB or 8GB RAM model)
- Raspberry Pi Camera Module or USB webcam
- MicroSD card (minimum 32GB recommended)
- Power supply

### Software Requirements

- Raspberry Pi OS (64-bit recommended)
- Python 3.8 or higher

## Installation

### 1. Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 2. Install System Dependencies

```bash
# Install OpenCV dependencies
sudo apt-get install -y python3-opencv libopencv-dev

# Install camera support (for Pi Camera Module)
sudo apt-get install -y python3-picamera2

# Install other dependencies
sudo apt-get install -y python3-pip python3-venv
```

### 3. Create Virtual Environment

```bash
# Navigate to project directory
cd elec537-asmt3
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** On Raspberry Pi, PyTorch installation might take a while. Consider using pre-built wheels:

```bash
# For Raspberry Pi OS 64-bit
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 5. Enable Camera (Raspberry Pi)

```bash
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable
# Reboot after enabling
```

## Running the Project

### Task 1: Real-time Object Detection

Run the detection script:

```bash
python src/task1_detection.py
```

Press 'q' to quit. The script will:

- Load YOLO11n model (downloads if needed)
- Capture live video from camera
- Perform real-time object detection
- Display performance metrics
- Show summary on exit

### Task 2: Model Optimization and Quantization

**Step 1:** Export quantized models

```bash
python src/task2.b_optimization.py
```

This exports models in FP32, FP16, and INT8 formats to the `models/` directory.

**Step 2:** Run benchmarks and comparison

```bash
python src/task2.c_comparison.py
```

This benchmarks each model and displays detailed performance metrics.

## Configuration

Edit `src/common/config.yaml` to customize:

- Camera settings (resolution, FPS)
- Model settings (confidence threshold, device)
- Quantization formats
- File paths

## Troubleshooting

**Camera Issues:** Ensure USB camera is plugged in (`ls /dev/video*`) or Pi Camera is enabled in `raspi-config`

**Memory Issues:** Use smaller resolution (320x240), close other apps, consider swap file

**Slow Performance:** Use YOLO11n, reduce resolution/FPS in config

**Import Errors:** Activate venv, reinstall requirements, check Python 3.8+

## Resources

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/)
- [Raspberry Pi Camera Documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)
