import os
import time

import cv2
import numpy as np
import psutil
import yaml
from pathlib import Path
from ultralytics import YOLO

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Install with: pip install onnxruntime")


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
        
        # Check for FP16 ONNX
        fp16_path = models_dir / f"{model_name}_fp16.onnx"
        if fp16_path.exists():
            exported_models['fp16'] = str(fp16_path)
        
        # Check for INT8 ONNX
        int8_path = models_dir / f"{model_name}_int8.onnx"
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
    
    def _preprocess_image_for_onnx(self, img):
        """Preprocess image for ONNX model input"""
        img_resized = cv2.resize(img, (320, 320))
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1] and convert to float32
        img_normalized = img_rgb.astype(np.float32) / 255.0
        # Transpose to CHW format
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)
        return img_batch
    
    def _count_detections_from_onnx(self, outputs, conf_threshold):
        """Count detections from ONNX model output"""
        # YOLO ONNX output format: [batch, num_predictions, 84]
        # 84 = 4 (bbox) + 80 (classes)
        predictions = outputs[0][0]  # Get first batch
        
        # Filter by confidence
        # Get max class confidence for each prediction
        class_confs = predictions[:, 4:]
        max_conf = np.max(class_confs, axis=1)
        
        # Count detections above threshold
        num_detections = np.sum(max_conf > conf_threshold)
        return int(num_detections)
    
    def benchmark_model(self, model_path, format_type, validation_images):
        print(f"Benchmarking {format_type}...", end=" ")
        
        try:
            # Get memory before loading model
            process = psutil.Process()
            mem_before_load = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Load model based on format
            model = None
            ort_session = None
            
            if format_type == "fp32":
                model = YOLO(model_path)
            elif format_type in ["fp16", "int8"]:
                # Use ONNX Runtime for quantized models
                if not ONNX_AVAILABLE:
                    print("Failed: onnxruntime not available")
                    return None
                
                # Configure ONNX Runtime session
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                ort_session = ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )
            else:
                model = YOLO(model_path)
            
            # Get memory after loading model
            mem_after_load = process.memory_info().rss / (1024 * 1024)  # MB
            model_memory = mem_after_load - mem_before_load
            
            inference_times = []
            memory_usages = []
            correct_detections = 0
            total_detections = 0
            
            # Warmup
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            if model:
                for _ in range(5):
                    _ = model(dummy_img, verbose=False)
            elif ort_session:
                dummy_input = self._preprocess_image_for_onnx(dummy_img)
                for _ in range(5):
                    _ = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_input})
            
            # Benchmark on validation images
            for img_path in validation_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Measure inference time and memory
                mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                start_time = time.time()
                
                if model:
                    # Use YOLO model for FP32
                    results = model(img, conf=self.model_config['confidence_threshold'], 
                                  verbose=False)
                    num_detections = len(results[0].boxes)
                elif ort_session:
                    # Use ONNX Runtime for quantized models
                    input_tensor = self._preprocess_image_for_onnx(img)
                    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
                    num_detections = self._count_detections_from_onnx(outputs, self.model_config['confidence_threshold'])
                else:
                    num_detections = 0
                
                inference_time = (time.time() - start_time) * 1000
                mem_after = process.memory_info().rss / (1024 * 1024)  # MB
                mem_usage = mem_after - mem_before
                
                inference_times.append(inference_time)
                memory_usages.append(mem_after)  # Track total memory in use
                
                # Count detections
                if num_detections > 0:
                    total_detections += num_detections
                    correct_detections += num_detections  # Simplified accuracy
            
            # Calculate metrics
            avg_inference_time = np.mean(inference_times) if inference_times else 0
            model_size = self.get_model_size(model_path)
            
            # Get CPU usage during inference
            cpu_usage = psutil.cpu_percent(interval=1.0)
            
            # Calculate memory metrics
            avg_memory_usage = np.mean(memory_usages) if memory_usages else 0
            peak_memory_usage = np.max(memory_usages) if memory_usages else 0
            
            metrics = {
                'format': format_type,
                'model_path': model_path,
                'model_size_mb': round(model_size, 2),
                'avg_inference_time_ms': round(avg_inference_time, 2),
                'min_inference_time_ms': round(np.min(inference_times), 2) if inference_times else 0,
                'max_inference_time_ms': round(np.max(inference_times), 2) if inference_times else 0,
                'avg_fps': round(1000 / avg_inference_time, 2) if avg_inference_time > 0 else 0,
                'cpu_usage_percent': round(cpu_usage, 2),
                'model_memory_mb': round(model_memory, 2),
                'avg_memory_mb': round(avg_memory_usage, 2),
                'peak_memory_mb': round(peak_memory_usage, 2),
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
            print(f"  Model Memory: {metrics['model_memory_mb']} MB")
            print(f"  Average Memory Usage: {metrics['avg_memory_mb']} MB")
            print(f"  Peak Memory Usage: {metrics['peak_memory_mb']} MB")
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
