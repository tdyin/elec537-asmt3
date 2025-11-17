# Description

The goal of this assignment is to gain hands-on experience with embedded AI on Raspberry Pi, focusing on model deployment and optimization techniques such as quantization and pruning. You will use a Raspberry Pi 4 with a camera module to run an on-device object detection model and study trade-offs between inference time, memory, and accuracy.

You have freedom in this assignment to use whichever tools or frameworks you prefer like Edge Impulse, TensorFlow Lite (now called LiteRT), or any other library of your choice, to collect datasets, train models, apply quantization/pruning, and evaluate performance.

## Task 1: Object Detection on Raspberry Pi

In this task, you will run a lightweight object detection model on your Raspberry Pi using the camera input.

**(a) Select a pre-trained lightweight model such as MobileNet-SSD, EfficientDet-Lite, or YOLOv5-Nano. You may use TensorFlow Lite (now LiteRT), PyTorch Lite, Edge Impulse, or any other library of your choice.**

**(b) Capture live video from the Raspberry Pi camera and perform object detection in real time. Display bounding boxes and class labels on the image frames. Include 2–3 example frames showing correct detections.**

**(c) To evaluate, measure and report:**

- Average inference time per frame (in milliseconds)
- Model size (in megabytes)
- CPU utilization (in percent)

Briefly describe your setup, including model type, framework, image resolution, and inference pipeline.

**(d) Record a short demonstration video (maximum 2 minutes) showing your Pi detecting at least three different objects in real time. Upload the video as unlisted on YouTube and include the link in your report.**

## Task 2: Model Optimization and Quantization

This task explores how model optimization affects inference time, memory, and accuracy.

**(a) Start with your original floating-point model from Task 1 and record its baseline metrics:**

- Model size (MB)
- Average inference time (ms)
- Accuracy on a small validation set (for example, 5–10 images per class)

**(b) Apply post-training quantization using any supported framework. Compare at least two variants such as float16 and int8 quantization. Include short code snippets and explain the steps.**

**(c) Deploy each quantized model on your Pi and measure the same metrics as above.**

**(d) Summarize your findings in a table comparing the original and quantized models. Discuss what trade-offs you observe in accuracy, latency, and memory usage, and which model gives the best overall balance for on-device use.**

**(e) Try an additional optimization such as pruning or knowledge distillation and briefly describe your results.**
