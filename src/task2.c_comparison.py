import os
import time

import cv2
import numpy as np
import psutil
import yaml
from pathlib import Path
from ultralytics import YOLO


class ModelComparison:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.task2_config = self.config['task2']
        self.results = {}
    
    def get_model_size(self, model_path):
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        return 0
       
    def get_exported_models(self):
        models_dir = Path(self.paths['models_dir'])
        model_name = self.model_config['name']
        
        exported_models = {}
        
        # Check for FP32 (original model)
        fp32_path = models_dir / f"{model_name}.pt"
        if fp32_path.exists():
            exported_models['fp32'] = str(fp32_path)
        
        # Check for FP16
        fp16_path = models_dir / f"{model_name}_fp16.torchscript"
        if fp16_path.exists():
            exported_models['fp16'] = str(fp16_path)
        
        # Check for INT8
        int8_path = models_dir / f"{model_name}_int8.torchscript"
        if int8_path.exists():
            exported_models['int8'] = str(int8_path)
        
        return exported_models
   
    def collect_validation_images(self):
        val_dir = Path(self.paths['validation_images'])
        images = list(val_dir.glob("*.jpg"))
        
        if len(images) == 0:
            print(f"\nNo validation images found in {val_dir}")
            return []
        
        return images
    
    def benchmark_model(self, model_path, format_type, validation_images):
        print(f"Benchmarking {format_type}...", end=" ")
        
        try:
            # Load model based on format
            if format_type == "fp32":
                model = YOLO(model_path)
            elif format_type == "fp16":
                model = YOLO(model_path, task='detect')
            elif format_type == "int8":
                # For ONNX models, use YOLO with ONNX backend
                model = YOLO(model_path, task='detect')
            else:
                model = YOLO(model_path)
            
            inference_times = []
            correct_detections = 0
            total_detections = 0
            
            # Warmup
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(5):
                _ = model(dummy_img, verbose=False)
            
            # Benchmark on validation images
            for img_path in validation_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Measure inference time
                start_time = time.time()
                results = model(img, conf=self.model_config['confidence_threshold'], 
                              verbose=False)
                inference_time = (time.time() - start_time) * 1000
                
                inference_times.append(inference_time)
                
                # Count detections
                if len(results[0].boxes) > 0:
                    total_detections += len(results[0].boxes)
                    correct_detections += len(results[0].boxes)  # Simplified accuracy
            
            # Calculate metrics
            avg_inference_time = np.mean(inference_times) if inference_times else 0
            model_size = self.get_model_size(model_path)
            
            # Get CPU usage during inference
            cpu_usage = psutil.cpu_percent(interval=1.0)
            
            metrics = {
                'format': format_type,
                'model_path': model_path,
                'model_size_mb': round(model_size, 2),
                'avg_inference_time_ms': round(avg_inference_time, 2),
                'min_inference_time_ms': round(np.min(inference_times), 2) if inference_times else 0,
                'max_inference_time_ms': round(np.max(inference_times), 2) if inference_times else 0,
                'avg_fps': round(1000 / avg_inference_time, 2) if avg_inference_time > 0 else 0,
                'cpu_usage_percent': round(cpu_usage, 2),
                'total_detections': total_detections,
                'images_tested': len(validation_images)
            }
            
            print(f"Done")
            
            return metrics
            
        except Exception as e:
            print(f"Failed: {e}")
            return None
 
    def run_benchmarks(self):
        print("\nBenchmarking models...")
        
        exported_models = self.get_exported_models()
        
        if not exported_models:
            print("Error: No exported models found. Run task2_optimization.py first.")
            return False
        
        # Collect validation images
        validation_images = self.collect_validation_images()
        
        if len(validation_images) == 0:
            print("Warning: No validation images found. Proceeding with basic measurements only.")
        
        # Benchmark each model
        for format_type, model_path in exported_models.items():
            metrics = self.benchmark_model(model_path, format_type, validation_images)
            if metrics:
                self.results[format_type] = metrics
        
        return True
    
    def display_detailed_metrics(self):
        if not self.results:
            return
        
        print("\n" + "="*80)
        print("DETAILED METRICS")
        print("="*80)
        
        for format_type, metrics in self.results.items():
            print(f"\n{format_type.upper()} Model:")
            print("-" * 40)
            print(f"  Model Path: {metrics['model_path']}")
            print(f"  Model Size: {metrics['model_size_mb']} MB")
            print(f"  Average Inference Time: {metrics['avg_inference_time_ms']} ms")
            print(f"  Min Inference Time: {metrics['min_inference_time_ms']} ms")
            print(f"  Max Inference Time: {metrics['max_inference_time_ms']} ms")
            print(f"  Average FPS: {metrics['avg_fps']}")
            print(f"  CPU Usage: {metrics['cpu_usage_percent']}%")
            print(f"  Total Detections: {metrics['total_detections']}")
            print(f"  Images Tested: {metrics['images_tested']}")
        
        print("="*80)
    
    def run_comparison(self):
        print("\nTASK 2: MODEL QUANTIZATION COMPARISON")
        print("="*60)
        
        # Check if we need to run benchmarks first
        if not self.results:
            if not self.run_benchmarks():
                return
        
        # Display detailed metrics
        self.display_detailed_metrics()


if __name__ == "__main__":
    comparison = ModelComparison()
    comparison.run_comparison()
