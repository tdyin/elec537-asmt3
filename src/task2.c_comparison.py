import os
import time
import json

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
    def __init__(self, config_path="src/config.yaml", coco_classes_path="data/coco_classes.json"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(coco_classes_path, 'r') as f:
            self.coco_classes = json.load(f)

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
    
    def _preprocess_image_for_onnx(self, img):
        """Preprocess image for ONNX model input"""
        img_resized = cv2.resize(img, (640, 640))
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
        detected_classes, _ = self._extract_detections_from_onnx(outputs, conf_threshold)
        return len(detected_classes)
    
    def _extract_detections_from_onnx(self, outputs, conf_threshold):
        """Extract class IDs and confidences from ONNX model output"""
        # YOLO11 ONNX output: [batch, 84, 8400] format
        # Need to transpose to [batch, 8400, 84]
        predictions = outputs[0]
        
        if len(predictions.shape) == 3 and predictions.shape[1] < predictions.shape[2]:
            # Transpose from [1, 84, 8400] to [1, 8400, 84]
            predictions = np.transpose(predictions, (0, 2, 1))
        
        predictions = predictions[0]  # Get first batch: [8400, 84]
        
        # Extract bounding boxes and scores
        # Format: [x, y, w, h, class0_score, class1_score, ..., class79_score]
        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]
        
        # Get max class score and class ID for each detection
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Filter by confidence threshold
        mask = max_scores > conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = class_ids[mask]
        
        # Apply basic NMS to remove duplicate detections
        if len(filtered_boxes) > 0:
            # Convert from center format to corner format for NMS
            x_center, y_center, width, height = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            boxes_corner = np.stack([x1, y1, x2, y2], axis=1)
            
            # Simple NMS implementation
            keep_indices = self._nms(boxes_corner, filtered_scores, iou_threshold=0.45)
            
            detected_classes = filtered_classes[keep_indices]
            detected_confs = filtered_scores[keep_indices]
        else:
            detected_classes = np.array([])
            detected_confs = np.array([])
        
        return detected_classes, detected_confs
    
    def _nms(self, boxes, scores, iou_threshold=0.45):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=int)
    
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
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            total_images = 0
            
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
                
                total_images += 1
                gt_label, gt_class_id = self.get_ground_truth(img_path)
                
                # Measure inference time and memory
                mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                start_time = time.time()
                
                detected_classes = []
                if model:
                    # Use YOLO model for FP32
                    results = model(img, conf=self.model_config['confidence_threshold'], 
                                  verbose=False)
                    num_detections = len(results[0].boxes)
                    detected_classes = results[0].boxes.cls.cpu().numpy().astype(int) if num_detections > 0 else []
                elif ort_session:
                    # Use ONNX Runtime for quantized models
                    input_tensor = self._preprocess_image_for_onnx(img)
                    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_tensor})
                    num_detections = self._count_detections_from_onnx(outputs, self.model_config['confidence_threshold'])
                    detected_classes, _ = self._extract_detections_from_onnx(outputs, self.model_config['confidence_threshold'])
                else:
                    num_detections = 0
                
                inference_time = (time.time() - start_time) * 1000
                mem_after = process.memory_info().rss / (1024 * 1024)  # MB
                mem_usage = mem_after - mem_before
                
                inference_times.append(inference_time)
                memory_usages.append(mem_after)  # Track total memory in use
                
                # Count detections
                total_detections += num_detections
                
                # Calculate accuracy metrics
                if gt_class_id is not None:
                    detected = gt_class_id in detected_classes
                    if detected:
                        true_positives += 1
                    else:
                        false_negatives += 1
                    
                    # Count false positives (detections of other classes)
                    other_detections = len([c for c in detected_classes if c != gt_class_id])
                    false_positives += other_detections
            
            # Calculate metrics
            avg_inference_time = np.mean(inference_times) if inference_times else 0
            model_size = self.get_model_size(model_path)
            
            # Get CPU usage during inference
            cpu_usage = psutil.cpu_percent(interval=1.0)
            
            # Calculate memory metrics
            avg_memory_usage = np.mean(memory_usages) if memory_usages else 0
            peak_memory_usage = np.max(memory_usages) if memory_usages else 0
            
            # Calculate accuracy metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = true_positives / total_images if total_images > 0 else 0
            
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
