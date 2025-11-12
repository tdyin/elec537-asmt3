# Report Template - ELEC537 Assignment 3

**Student Name:** [Your Name]  
**Student ID:** [Your ID]  
**Date:** [Submission Date]

---

## Task 1: Object Detection on Raspberry Pi

### Setup Description

**Hardware:**
- Raspberry Pi Model: [e.g., Raspberry Pi 4 - 4GB RAM]
- Camera: [e.g., Raspberry Pi Camera Module 3 / USB Webcam]
- Power Supply: [specifications]

**Software:**
- Operating System: [e.g., Raspberry Pi OS 64-bit]
- Python Version: [e.g., 3.9.2]
- Framework: [e.g., PyTorch 2.0.0, Ultralytics YOLO11]

**Model Configuration:**
- Model Type: [e.g., YOLO11-Nano]
- Image Resolution: [e.g., 640x480]
- Confidence Threshold: [e.g., 0.5]
- Inference Pipeline: [describe preprocessing, inference, postprocessing]

### Performance Metrics

| Metric | Value |
|--------|-------|
| Average Inference Time | ___ ms |
| Min Inference Time | ___ ms |
| Max Inference Time | ___ ms |
| Average FPS | ___ fps |
| Model Size | ___ MB |
| Average CPU Utilization | ___ % |

### Example Detections

**Frame 1:**
![Detection Example 1](../results/captured_frames/detection_1.jpg)
- Detected Objects: [list objects with confidence scores]
- Inference Time: ___ ms

**Frame 2:**
![Detection Example 2](../results/captured_frames/detection_2.jpg)
- Detected Objects: [list objects with confidence scores]
- Inference Time: ___ ms

**Frame 3:**
![Detection Example 3](../results/captured_frames/detection_3.jpg)
- Detected Objects: [list objects with confidence scores]
- Inference Time: ___ ms

### Demonstration Video

**Video Link:** [YouTube URL]

**Description:** [Brief description of what's shown in the video - objects detected, environment, duration]

### Observations and Analysis

[Discuss your observations:]
- How well did the model perform?
- Were there any challenging scenarios (lighting, distance, occlusion)?
- What was the user experience like (responsiveness, accuracy)?
- Any limitations or challenges encountered?

---

## Task 2: Model Optimization and Quantization

### Baseline Model (FP32)

**Baseline Metrics:**

| Metric | Value |
|--------|-------|
| Model Size | ___ MB |
| Average Inference Time | ___ ms |
| Average FPS | ___ fps |
| Accuracy (Validation Set) | ___% |
| CPU Utilization | ___% |

**Validation Set:**
- Number of Images: ___
- Classes Represented: [list classes]
- Image Resolution: ___x___

### Quantization Methods

#### FP16 Quantization

**Code Snippet:**
```python
# Include your quantization code here
model = YOLO('yolo11n.pt')
model.export(format='torchscript', half=True)
```

**Explanation:**
[Explain the quantization process, what FP16 means, and how you applied it]

**Results:**

| Metric | Value |
|--------|-------|
| Model Size | ___ MB |
| Average Inference Time | ___ ms |
| Average FPS | ___ fps |
| Accuracy (Validation Set) | ___% |
| CPU Utilization | ___% |

#### INT8 Quantization

**Code Snippet:**
```python
# Include your quantization code here
model = YOLO('yolo11n.pt')
model.export(format='onnx', int8=True, simplify=True)
```

**Explanation:**
[Explain the quantization process, what INT8 means, and how you applied it]

**Results:**

| Metric | Value |
|--------|-------|
| Model Size | ___ MB |
| Average Inference Time | ___ ms |
| Average FPS | ___ fps |
| Accuracy (Validation Set) | ___% |
| CPU Utilization | ___% |

### Comparison Table

| Model Variant | Size (MB) | Inference (ms) | FPS | Accuracy | CPU % | Size Reduction | Speed Improvement |
|---------------|-----------|----------------|-----|----------|-------|----------------|-------------------|
| FP32 (Baseline) | ___ | ___ | ___ | ___% | ___% | 0% | 0% |
| FP16 | ___ | ___ | ___ | ___% | ___% | ___% | ___% |
| INT8 | ___ | ___ | ___ | ___% | ___% | ___% | ___% |

### Trade-off Analysis

**Model Size:**
[Discuss the reduction in model size across variants]

**Inference Speed:**
[Discuss changes in inference time and FPS]

**Accuracy:**
[Discuss any accuracy degradation observed]

**Overall Balance:**
[Which model provides the best balance for on-device deployment? Why?]

### Additional Optimization (Optional)

**Method:** [e.g., Pruning, Knowledge Distillation, Neural Architecture Search]

**Code Snippet:**
```python
# Include code for additional optimization
```

**Results:**

| Metric | Value |
|--------|-------|
| Model Size | ___ MB |
| Average Inference Time | ___ ms |
| Average FPS | ___ fps |
| Accuracy | ___% |

**Observations:**
[Describe results and effectiveness of the additional optimization]

---

## Conclusion

### Key Findings

1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### Challenges and Solutions

**Challenge 1:** [Description]
- **Solution:** [How you addressed it]

**Challenge 2:** [Description]
- **Solution:** [How you addressed it]

### Recommendations

[Based on your experiments, what recommendations would you make for deploying object detection on edge devices?]

### Future Improvements

[What could be improved or explored further?]

---

## Appendix

### System Information

```
CPU: [specifications]
RAM: [amount]
Storage: [type and capacity]
OS: [version]
Python: [version]
PyTorch: [version]
```

### Additional Resources

- Code Repository: [GitHub link if applicable]
- Additional Images: [folder location]
- Raw Data: [location of raw performance logs]

---

**Declaration:** I certify that this work is my own and that I have properly cited all sources used in this report.

**Signature:** ________________  
**Date:** ________________
