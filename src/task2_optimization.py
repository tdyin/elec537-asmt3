import time
import yaml
import cv2
import os
import json
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import torch


class ModelOptimizer:
    def __init__(self, config_path="config.yaml"):
        """Initialize the model optimizer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.task2_config = self.config['task2']
        
        # Create necessary directories
        Path(self.paths['models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.paths['validation_images']).mkdir(parents=True, exist_ok=True)
        Path(f"{self.paths['results_dir']}/task2").mkdir(parents=True, exist_ok=True)
        
        # Load base model
        print(f"Loading base {self.model_config['name']} model...")
        self.base_model_path = f"{self.paths['models_dir']}/{self.model_config['name']}.pt"
        
        if not os.path.exists(self.base_model_path):
            print("Model not found locally. Downloading...")
            model = YOLO(self.model_config['name'])
            model.save(self.base_model_path)
        
        self.results = {}
        
    def get_model_size(self, model_path):
        """Get model size in MB."""
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        return 0
    
    def export_quantized_models(self):
        """Export models in different quantization formats."""
        print("\n" + "="*60)
        print("EXPORTING QUANTIZED MODELS")
        print("="*60)
        
        model = YOLO(self.base_model_path)
        exported_models = {}
        
        for format_type in self.task2_config['quantization']['formats']:
            print(f"\nExporting {format_type} model...")
            
            try:
                if format_type == "fp32":
                    # Baseline - just copy the original PyTorch model
                    exported_models[format_type] = self.base_model_path
                    print(f"Using original model: {self.base_model_path}")
                
                elif format_type == "fp16":
                    # Export to TorchScript with FP16
                    export_path = f"{self.paths['models_dir']}/{self.model_config['name']}_fp16.torchscript"
                    model.export(format='torchscript', half=True)
                    # Rename to expected location
                    default_export = self.base_model_path.replace('.pt', '.torchscript')
                    if os.path.exists(default_export):
                        os.rename(default_export, export_path)
                    exported_models[format_type] = export_path
                    print(f"Exported FP16 model: {export_path}")
                
                elif format_type == "int8":
                    # Export to ONNX with INT8 quantization
                    export_path = f"{self.paths['models_dir']}/{self.model_config['name']}_int8.onnx"
                    model.export(format='onnx', int8=True, simplify=True)
                    # Rename to expected location
                    default_export = self.base_model_path.replace('.pt', '.onnx')
                    if os.path.exists(default_export):
                        os.rename(default_export, export_path)
                    exported_models[format_type] = export_path
                    print(f"Exported INT8 model: {export_path}")
                
            except Exception as e:
                print(f"Error exporting {format_type}: {e}")
        
        return exported_models
    
    def benchmark_model(self, model_path, format_type, validation_images):
        """Benchmark a single model variant."""
        print(f"\nBenchmarking {format_type} model...")
        
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
            
            print(f"Results for {format_type}:")
            print(f"  Model Size: {metrics['model_size_mb']} MB")
            print(f"  Avg Inference: {metrics['avg_inference_time_ms']} ms")
            print(f"  FPS: {metrics['avg_fps']}")
            
            return metrics
            
        except Exception as e:
            print(f"Error benchmarking {format_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def collect_validation_images(self):
        """Collect or generate validation images."""
        val_dir = Path(self.paths['validation_images'])
        images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
        
        if len(images) == 0:
            print(f"\nNo validation images found in {val_dir}")
            print("Please add validation images to the directory or capture them using task1_detection.py")
            print("For now, using test images from COCO dataset...")
            
            # Download some sample images if needed
            # For now, return empty list
            return []
        
        target_count = min(len(images), self.task2_config['validation_set_size'])
        return images[:target_count]
    
    def run_comparison(self):
        """Run complete quantization comparison."""
        print("\n" + "="*60)
        print("TASK 2: MODEL OPTIMIZATION AND QUANTIZATION")
        print("="*60)
        
        # Step 1: Export quantized models
        exported_models = self.export_quantized_models()
        
        # Step 2: Collect validation images
        validation_images = self.collect_validation_images()
        
        if len(validation_images) == 0:
            print("\nWarning: No validation images available for accuracy testing.")
            print("Proceeding with inference time and model size measurements only.")
        
        # Step 3: Benchmark each model
        print("\n" + "="*60)
        print("BENCHMARKING MODELS")
        print("="*60)
        
        for format_type, model_path in exported_models.items():
            metrics = self.benchmark_model(model_path, format_type, validation_images)
            if metrics:
                self.results[format_type] = metrics
        
        # Step 4: Save and display results
        self.save_results()
        self.display_comparison_table()
    
    def save_results(self):
        """Save results to JSON file."""
        results_file = f"{self.paths['results_dir']}/task2/quantization_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    def display_comparison_table(self):
        """Display comparison table of all models."""
        print("\n" + "="*80)
        print("QUANTIZATION COMPARISON TABLE")
        print("="*80)
        
        # Header
        header = f"{'Format':<10} | {'Size (MB)':<10} | {'Inference (ms)':<15} | {'FPS':<8} | {'CPU %':<8}"
        print(header)
        print("-" * len(header))
        
        # Baseline first
        if 'fp32' in self.results:
            r = self.results['fp32']
            print(f"{r['format']:<10} | {r['model_size_mb']:<10.2f} | {r['avg_inference_time_ms']:<15.2f} | "
                  f"{r['avg_fps']:<8.2f} | {r['cpu_usage_percent']:<8.2f}")
        
        # Other formats
        for format_type in ['fp16', 'int8']:
            if format_type in self.results:
                r = self.results[format_type]
                print(f"{r['format']:<10} | {r['model_size_mb']:<10.2f} | {r['avg_inference_time_ms']:<15.2f} | "
                      f"{r['avg_fps']:<8.2f} | {r['cpu_usage_percent']:<8.2f}")
        
        print("="*80)
        
        # Calculate improvements
        if 'fp32' in self.results:
            baseline = self.results['fp32']
            print("\nIMPROVEMENTS OVER BASELINE (FP32):")
            print("-" * 60)
            
            for format_type in ['fp16', 'int8']:
                if format_type in self.results:
                    r = self.results[format_type]
                    size_reduction = ((baseline['model_size_mb'] - r['model_size_mb']) / 
                                     baseline['model_size_mb'] * 100)
                    speed_improvement = ((baseline['avg_inference_time_ms'] - r['avg_inference_time_ms']) / 
                                        baseline['avg_inference_time_ms'] * 100)
                    
                    print(f"\n{format_type.upper()}:")
                    print(f"  Size Reduction: {size_reduction:.1f}%")
                    print(f"  Speed Improvement: {speed_improvement:.1f}%")
        
        print("="*80)


if __name__ == "__main__":
    optimizer = ModelOptimizer()
    optimizer.run_comparison()
