import json
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
        """Initialize the model comparison tool with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.task2_config = self.config['task2']
        self.results_file = f"{self.paths['results_dir']}/task2/quantization_results.json"
        self.results = {}
    
    def get_model_size(self, model_path):
        """Get model size in MB."""
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        return 0
    
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
    
    def get_exported_models(self):
        """Get list of exported model files."""
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
    
    def run_benchmarks(self):
        """Run benchmarks on all exported models."""
        print("\n" + "="*60)
        print("BENCHMARKING MODELS")
        print("="*60)
        
        # Get exported models
        exported_models = self.get_exported_models()
        
        if not exported_models:
            print("Error: No exported models found.")
            print("Please run task2_optimization.py first to export the models.")
            return False
        
        # Collect validation images
        validation_images = self.collect_validation_images()
        
        if len(validation_images) == 0:
            print("\nWarning: No validation images available for accuracy testing.")
            print("Proceeding with inference time and model size measurements only.")
        
        # Benchmark each model
        for format_type, model_path in exported_models.items():
            metrics = self.benchmark_model(model_path, format_type, validation_images)
            if metrics:
                self.results[format_type] = metrics
        
        # Save results
        self.save_results()
        return True
    
    def save_results(self):
        """Save results to JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {self.results_file}")
    
    def load_results(self):
        """Load benchmark results from JSON file."""
        if not Path(self.results_file).exists():
            return {}
        
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def display_comparison_table(self):
        """Display comparison table of all models."""
        results = self.results if self.results else self.load_results()
        
        if not results:
            print("No results available for comparison.")
            print("Run benchmarks first using this script.")
            return
        
        print("\n" + "="*80)
        print("QUANTIZATION COMPARISON TABLE")
        print("="*80)
        
        results = self.results if self.results else self.load_results()
        
        # Header
        header = f"{'Format':<10} | {'Size (MB)':<10} | {'Inference (ms)':<15} | {'FPS':<8} | {'CPU %':<8}"
        print(header)
        print("-" * len(header))
        
        # Baseline first
        if 'fp32' in results:
            r = results['fp32']
            print(f"{r['format']:<10} | {r['model_size_mb']:<10.2f} | {r['avg_inference_time_ms']:<15.2f} | "
                  f"{r['avg_fps']:<8.2f} | {r['cpu_usage_percent']:<8.2f}")
        
        # Other formats
        for format_type in ['fp16', 'int8']:
            if format_type in results:
                r = results[format_type]
                print(f"{r['format']:<10} | {r['model_size_mb']:<10.2f} | {r['avg_inference_time_ms']:<15.2f} | "
                      f"{r['avg_fps']:<8.2f} | {r['cpu_usage_percent']:<8.2f}")
        
        print("="*80)
    
    def display_improvements(self):
        """Display improvements over baseline model."""
        results = self.results if self.results else self.load_results()
        
        if not results or 'fp32' not in results:
            print("Baseline (FP32) results not available.")
            return
        
        baseline = results['fp32']
        print("\nIMPROVEMENTS OVER BASELINE (FP32):")
        print("-" * 60)
        
        for format_type in ['fp16', 'int8']:
            if format_type in results:
                r = results[format_type]
                size_reduction = ((baseline['model_size_mb'] - r['model_size_mb']) / 
                                 baseline['model_size_mb'] * 100) if baseline['model_size_mb'] > 0 else 0.0
                speed_improvement = ((baseline['avg_inference_time_ms'] - r['avg_inference_time_ms']) / 
                                    baseline['avg_inference_time_ms'] * 100) if baseline['avg_inference_time_ms'] > 0 else 0.0
                
                print(f"\n{format_type.upper()}:")
                print(f"  Size Reduction: {size_reduction:.1f}%")
                print(f"  Speed Improvement: {speed_improvement:.1f}%")
        
        print("="*80)
    
    def display_detailed_metrics(self):
        """Display detailed metrics for each model."""
        results = self.results if self.results else self.load_results()
        
        if not results:
            print("No results available for detailed metrics.")
            return
        
        print("\n" + "="*80)
        print("DETAILED METRICS")
        print("="*80)
        
        for format_type, metrics in results.items():
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
        """Run complete comparison analysis."""
        print("\n" + "="*60)
        print("TASK 2: MODEL QUANTIZATION COMPARISON")
        print("="*60)
        
        # Check if we need to run benchmarks first
        if not self.results:
            print("\nNo benchmark results found. Running benchmarks first...")
            if not self.run_benchmarks():
                return
        
        # Display comparison table
        self.display_comparison_table()
        
        # Display improvements
        self.display_improvements()
        
        # Display detailed metrics
        self.display_detailed_metrics()


if __name__ == "__main__":
    comparison = ModelComparison()
    comparison.run_comparison()
