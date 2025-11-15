import os
import time
import json
import cv2
import numpy as np
import psutil
from pathlib import Path
from common import load_config, get_model_size, calculate_metrics
from common import ModelHandler

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Install with: pip install onnxruntime")


class ModelComparison:
    def __init__(self, config_path="common/config.yaml", coco_classes_path="data/coco_classes.json"):
        self.config = load_config(config_path)
        
        with open(coco_classes_path, 'r') as f:
            self.coco_classes = json.load(f)

        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.task2_config = self.config['task2']
        self.results = {}
       
    def get_exported_models(self):
        models_dir = Path(self.paths['models_dir'])
        model_name = self.model_config['name']
        
        exported_models = {}
        
        for fmt, path_suffix in [
            ('fp32', '.pt'), ('fp16', '_fp16.onnx'), ('int8', '_int8.onnx')
        ]:
            model_path = models_dir / f"{model_name}{path_suffix}"
            if model_path.exists():
                exported_models[fmt] = str(model_path)
        
        return exported_models
   
    def collect_validation_images(self):
        val_dir = Path(self.paths['validation_images'])
        # Collect images from all subdirectories
        images = []
        if val_dir.exists():
            for subfolder in val_dir.iterdir():
                if subfolder.is_dir():
                    images.extend(list(subfolder.glob("*.jpg")))
        
        if len(images) == 0:
            print(f"\nNo validation images found in {val_dir}")
            return []
        
        return images
    
    def get_ground_truth(self, img_path):
        """Extract ground truth from parent folder name"""
        folder_name = Path(img_path).parent.name.lower()
        class_id = self.coco_classes.get(folder_name)
        
        if class_id is not None:
            return folder_name, class_id
        
        return None, None
    
    def _extract_detections(self, model_handler, results):
        """Extract detected classes from model results"""
        if model_handler.model:
            # PyTorch model
            num_detections = len(results[0].boxes)
            detected_classes = results[0].boxes.cls.cpu().numpy().astype(int) if num_detections > 0 else []
        else:
            # ONNX model
            detected_classes, _ = results
        return detected_classes
    
    def benchmark_model(self, model_path, format_type, validation_images):
        print(f"Benchmarking {format_type}...", end=" ")
        
        try:
            process = psutil.Process()
            mem_before_load = process.memory_info().rss / (1024 * 1024)
            
            # Load model using ModelHandler
            model_handler = ModelHandler(model_path, format_type)
            
            mem_after_load = process.memory_info().rss / (1024 * 1024)
            model_memory = mem_after_load - mem_before_load
            
            inference_times = []
            memory_usages = []
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            total_images = 0
            total_detections = 0
            
            # Warmup
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(5):
                _ = model_handler.predict(dummy_img, verbose=False)
            
            # Benchmark on validation images
            for img_path in validation_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                total_images += 1
                gt_label, gt_class_id = self.get_ground_truth(img_path)
                
                mem_before = process.memory_info().rss / (1024 * 1024)
                start_time = time.time()
                
                results = model_handler.predict(
                    img, conf_threshold=self.model_config['confidence_threshold'], verbose=False
                )
                
                inference_time = (time.time() - start_time) * 1000
                mem_after = process.memory_info().rss / (1024 * 1024)
                
                inference_times.append(inference_time)
                memory_usages.append(mem_after)
                
                detected_classes = self._extract_detections(model_handler, results)
                total_detections += len(detected_classes)
                
                # Calculate accuracy metrics
                if gt_class_id is not None:
                    if gt_class_id in detected_classes:
                        true_positives += 1
                    else:
                        false_negatives += 1
                    other_detections = len([c for c in detected_classes if c != gt_class_id])
                    false_positives += other_detections
            
            # Calculate metrics
            min_inf, max_inf, avg_inf = calculate_metrics(inference_times)
            cpu_usage = psutil.cpu_percent(interval=1.0)
            _, peak_mem, avg_mem = calculate_metrics(memory_usages)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = true_positives / total_images if total_images > 0 else 0
            
            metrics = {
                'format': format_type,
                'model_path': model_path,
                'model_size_mb': round(get_model_size(model_path), 2),
                'avg_inference_time_ms': round(avg_inf, 2),
                'min_inference_time_ms': round(min_inf, 2),
                'max_inference_time_ms': round(max_inf, 2),
                'avg_fps': round(1000 / avg_inf, 2) if avg_inf > 0 else 0,
                'cpu_usage_percent': round(cpu_usage, 2),
                'model_memory_mb': round(model_memory, 2),
                'avg_memory_mb': round(avg_mem, 2),
                'peak_memory_mb': round(peak_mem, 2),
                'total_detections': total_detections,
                'images_tested': len(validation_images),
                'accuracy': round(accuracy * 100, 2),
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1_score': round(f1_score * 100, 2),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
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
            print(f"\n  Accuracy Metrics:")
            print(f"    Accuracy: {metrics['accuracy']}%")
            print(f"    Precision: {metrics['precision']}%")
            print(f"    Recall: {metrics['recall']}%")
            print(f"    F1 Score: {metrics['f1_score']}%")
            print(f"    True Positives: {metrics['true_positives']}")
            print(f"    False Positives: {metrics['false_positives']}")
            print(f"    False Negatives: {metrics['false_negatives']}")
        
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
