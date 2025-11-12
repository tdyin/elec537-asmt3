# ELEC537 Assignment 3 - Setup Guide

## Project Structure
```
elec537-asmt3/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── README.md               # Assignment description
├── src/                    # Source code
│   ├── task1_detection.py # Real-time object detection
│   ├── task2_optimization.py # Model quantization
│   ├── utils.py           # Utility functions
│   └── performance.py     # Performance monitoring
├── models/                 # Model files (downloaded automatically)
├── data/                   # Data directory
│   └── validation/        # Validation images for Task 2
├── results/                # Results and outputs
│   ├── captured_frames/   # Saved detection frames
│   ├── videos/            # Recorded videos
│   └── task2/             # Quantization results
└── docs/                   # Documentation
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
cd /Users/yin/Dev/elec537-asmt3
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

1. **Run the detection script:**
```bash
cd src
python task1_detection.py
```

2. **Controls:**
   - Press `s` to save the current frame
   - Press `q` to quit

3. **What it does:**
   - Loads YOLO11n model (downloads if not present)
   - Captures live video from camera
   - Performs real-time object detection
   - Displays bounding boxes and labels
   - Shows performance metrics on screen
   - Saves example frames to `results/captured_frames/`

4. **Expected Output:**
   - Live video window with detections
   - Console showing performance metrics
   - Summary report when exiting

### Task 2: Model Optimization and Quantization

1. **Prepare validation images:**
   - Add 50+ images to `data/validation/` directory
   - Or use frames captured from Task 1

2. **Run the optimization script:**
```bash
cd src
python task2_optimization.py
```

3. **What it does:**
   - Exports models in FP32, FP16, and INT8 formats
   - Benchmarks each model variant
   - Measures inference time, model size, CPU usage
   - Generates comparison table
   - Saves results to `results/task2/quantization_results.json`

4. **Expected Output:**
   - Quantized models saved to `models/` directory
   - Performance comparison table in console
   - JSON file with detailed metrics

## Configuration

Edit `config.yaml` to customize:

- **Camera settings:** Resolution, FPS, rotation
- **Model settings:** Model variant, confidence threshold
- **Paths:** Output directories
- **Quantization formats:** Which formats to test

## Troubleshooting

### Camera Issues
- **USB Camera:** Ensure it's plugged in and recognized (`ls /dev/video*`)
- **Pi Camera:** Verify it's enabled in `raspi-config`
- **Test camera:** `libcamera-hello` (for Pi Camera Module 3)

### Memory Issues
- Use smaller image resolution (e.g., 416x416)
- Close other applications
- Consider using swap file if RAM is limited

### Slow Performance
- YOLO11n (nano) is the fastest variant
- Reduce camera resolution
- Lower FPS in config
- Ensure other processes aren't consuming CPU

### Import Errors
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
- Check Python version: `python --version` (3.8+ required)

## Tips for Success

1. **Start Simple:** Test with USB webcam first before switching to Pi Camera
2. **Good Lighting:** Ensure adequate lighting for better detection accuracy
3. **Stable Mount:** Use a tripod or stable surface for the camera
4. **Multiple Objects:** Test with variety of common objects (cup, phone, keyboard, etc.)
5. **Record Video:** Use OBS Studio or `ffmpeg` to record screen for demonstration

## Next Steps

1. Run Task 1 and capture example frames
2. Record demonstration video (max 2 minutes)
3. Collect validation images for Task 2
4. Run quantization experiments
5. Analyze results and write report

## Resources

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/)
- [PyTorch Mobile Documentation](https://pytorch.org/mobile/)
- [Raspberry Pi Camera Documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [TensorFlow Lite (LiteRT)](https://www.tensorflow.org/lite)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the assignment description in `README.md`
3. Consult course materials and documentation
