"""
Task 1: Real-time Object Detection on Raspberry Pi
Runs YOLO11 model on live camera feed and measures performance metrics.
"""

import time
import yaml
import cv2
import psutil
import os
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import numpy as np


class ObjectDetector:
    def __init__(self, config_path="config.yaml"):
        """Initialize the object detector with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.camera_config = self.config['camera']
        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.task1_config = self.config['task1']
        self.monitoring = self.config['monitoring']
        
        # Create necessary directories
        Path(self.paths['captured_frames']).mkdir(parents=True, exist_ok=True)
        Path(self.paths['models_dir']).mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"Loading {self.model_config['name']} model...")
        model_path = f"{self.paths['models_dir']}/{self.model_config['name']}.pt"
        
        if not os.path.exists(model_path):
            print("Model not found locally. Downloading...")
            self.model = YOLO(self.model_config['name'])
            self.model.save(model_path)
        else:
            self.model = YOLO(model_path)
        
        print("Model loaded successfully!")
        
        # Performance metrics
        self.inference_times = []
        self.cpu_usages = []
        self.frame_count = 0
        self.saved_frames = 0
        
    def initialize_camera(self):
        """Initialize camera (USB or Pi Camera)."""
        try:
            # Try USB camera first (for development on non-Pi machines)
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            # Set camera resolution
            width, height = self.camera_config['resolution']
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_config['fps'])
            
            print(f"Camera initialized: {width}x{height} @ {self.camera_config['fps']}fps")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def get_cpu_usage(self):
        """Get current CPU utilization."""
        return psutil.cpu_percent(interval=0.1)
    
    def get_model_size(self):
        """Get model size in MB."""
        model_path = f"{self.paths['models_dir']}/{self.model_config['name']}.pt"
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        return 0
    
    def process_frame(self, frame):
        """Run object detection on a single frame."""
        start_time = time.time()
        
        # Get image size from config (default 320 for faster inference)
        imgsz = self.model_config.get('imgsz', 320)
        
        # Run inference with optimizations
        results = self.model(
            frame,
            conf=self.model_config['confidence_threshold'],
            iou=self.model_config['iou_threshold'],
            device=self.model_config['device'],
            imgsz=imgsz,  # Use smaller image size for speed
            verbose=False,
            half=False,  # Disable FP16 on CPU (not supported)
            augment=False,  # Disable test-time augmentation
            max_det=100  # Limit max detections for speed
        )
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        return annotated_frame, inference_time, results[0]
    
    def save_frame(self, frame, detections):
        """Save annotated frame to disk."""
        if self.saved_frames < self.task1_config['max_frames_to_save']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.paths['captured_frames']}/detection_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.saved_frames += 1
            print(f"Saved frame: {filename} (Detections: {len(detections.boxes)})")
    
    def display_metrics(self, frame, inference_time, cpu_usage):
        """Overlay performance metrics on frame."""
        avg_inference = np.mean(self.inference_times[-30:]) if self.inference_times else 0
        fps = 1000 / avg_inference if avg_inference > 0 else 0
        
        # Create metrics text
        metrics = [
            f"Inference: {inference_time:.1f}ms (Avg: {avg_inference:.1f}ms)",
            f"FPS: {fps:.1f}",
            f"CPU: {cpu_usage:.1f}%",
            f"Frame: {self.frame_count}"
        ]
        
        # Draw metrics on frame
        y_offset = 30
        for metric in metrics:
            cv2.putText(frame, metric, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
        
        return frame
    
    def run(self):
        """Main detection loop."""
        if not self.initialize_camera():
            print("Failed to initialize camera. Exiting.")
            return
        
        print("\n" + "="*60)
        print("Starting object detection...")
        print("Press 's' to save current frame")
        print("Press 'q' to quit")
        print("="*60 + "\n")
        
        # Frame skipping for better performance
        frame_skip = 0  # Process every frame initially
        skip_counter = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                self.frame_count += 1
                skip_counter += 1
                
                # Skip frames if needed (process every Nth frame)
                if skip_counter <= frame_skip:
                    continue
                skip_counter = 0
                
                # Process frame
                annotated_frame, inference_time, results = self.process_frame(frame)
                
                # Get CPU usage (less frequently to save cycles)
                if self.frame_count % 5 == 0:
                    cpu_usage = self.get_cpu_usage()
                else:
                    cpu_usage = 0
                
                # Record metrics after warmup
                if self.frame_count > self.monitoring['warmup_frames']:
                    self.inference_times.append(inference_time)
                    if cpu_usage > 0:  # Only record when actually measured
                        self.cpu_usages.append(cpu_usage)
                
                # Display metrics
                display_frame = self.display_metrics(annotated_frame, inference_time, cpu_usage)
                
                # Show frame
                if self.task1_config['display_preview']:
                    cv2.imshow('Object Detection', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and self.task1_config['save_frames']:
                    self.save_frame(annotated_frame, results)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
            self.print_summary()
    
    def cleanup(self):
        """Release resources."""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
    
    def print_summary(self):
        """Print performance summary."""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        if self.inference_times:
            avg_inference = np.mean(self.inference_times)
            min_inference = np.min(self.inference_times)
            max_inference = np.max(self.inference_times)
            
            print(f"Average Inference Time: {avg_inference:.2f} ms")
            print(f"Min Inference Time: {min_inference:.2f} ms")
            print(f"Max Inference Time: {max_inference:.2f} ms")
            print(f"Average FPS: {1000/avg_inference:.2f}")
        
        if self.cpu_usages:
            avg_cpu = np.mean(self.cpu_usages)
            min_cpu = np.min(self.cpu_usages)
            max_cpu = np.max(self.cpu_usages)
            
            print(f"Average CPU Utilization: {avg_cpu:.1f}%")
            print(f"Min CPU Utilization: {min_cpu:.1f}%")
            print(f"Max CPU Utilization: {max_cpu:.1f}%")
        
        model_size = self.get_model_size()
        print(f"Model Size: {model_size:.2f} MB")
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Frames Saved: {self.saved_frames}")
        print("="*60)


if __name__ == "__main__":
    detector = ObjectDetector()
    detector.run()
