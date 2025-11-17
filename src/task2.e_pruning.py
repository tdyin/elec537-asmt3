import os
import time
import json
import cv2
import numpy as np
import psutil
import torch
import torch.nn.utils.prune as prune
from pathlib import Path
from ultralytics import YOLO
from common import load_config, get_model_size, calculate_metrics

class ModelPruning:
    def __init__(self, config_path="src/config.yaml", coco_classes_path="data/coco_classes.json"):
        self.config = load_config(config_path)
        
        with open(coco_classes_path, 'r') as f:
            self.coco_classes = json.load(f)

        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.results = {}
       
    def prune_model(self, model_path, pruning_amount=0.3):
        """Apply magnitude-based pruning to YOLO model"""
        print(f"Pruning model with amount {pruning_amount}...", end=" ")
        
        try:
            model = YOLO(model_path)
            
            # Apply global unstructured pruning (more stable than structured)
            parameters_to_prune = []
            for name, module in model.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, 'weight'))
            
            # Global magnitude pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_amount,
            )
            
            # Make pruning permanent
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
            
            # Save pruned model
            models_dir = Path(self.paths['models_dir'])
            pruned_path = models_dir / f"{self.model_config['name']}_pruned.pt"
            model.save(str(pruned_path))
            
            print(f"Done -> {pruned_path}")
            return str(pruned_path)
            
        except Exception as e:
            print(f"Failed: {e}")
            return None
    
    def collect_validation_images(self):
        val_dir = Path(self.paths['validation_images'])
        images = []
        if val_dir.exists():
            for subfolder in val_dir.iterdir():
                if subfolder.is_dir():
                    images.extend(list(subfolder.glob("*.jpg")))
        return images if images else []
    
    def get_ground_truth(self, img_path):
        folder_name = Path(img_path).parent.name.lower()
        class_id = self.coco_classes.get(folder_name)
        return (folder_name, class_id) if class_id is not None else (None, None)
    
    def benchmark_model(self, model_path, model_name, validation_images):
        print(f"Benchmarking {model_name}...", end=" ")
        
        try:
            process = psutil.Process()
            model = YOLO(model_path)
            
            inference_times, memory_usages = [], []
            true_positives, false_positives, false_negatives = 0, 0, 0
            total_images, total_detections = 0, 0
            
            # Warmup
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(5):
                _ = model(dummy_img, conf=self.model_config['confidence_threshold'], verbose=False)
            
            for img_path in validation_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                total_images += 1
                gt_label, gt_class_id = self.get_ground_truth(img_path)
                
                mem_before = process.memory_info().rss / (1024 * 1024)
                start_time = time.time()
                
                results = model(img, conf=self.model_config['confidence_threshold'], verbose=False)
                
                inference_time = (time.time() - start_time) * 1000
                mem_after = process.memory_info().rss / (1024 * 1024)
                
                inference_times.append(inference_time)
                memory_usages.append(mem_after)
                
                detected_classes = results[0].boxes.cls.cpu().numpy().astype(int) if len(results[0].boxes) > 0 else []
                total_detections += len(detected_classes)
                
                if gt_class_id is not None:
                    if gt_class_id in detected_classes:
                        true_positives += 1
                    else:
                        false_negatives += 1
                    false_positives += len([c for c in detected_classes if c != gt_class_id])
            
            min_inf, max_inf, avg_inf = calculate_metrics(inference_times)
            _, peak_mem, avg_mem = calculate_metrics(memory_usages)
            cpu_usage = psutil.cpu_percent(interval=1.0)
            
            denom_p = true_positives + false_positives
            denom_r = true_positives + false_negatives
            precision = true_positives / denom_p if denom_p > 0 else 0
            recall = true_positives / denom_r if denom_r > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = true_positives / total_images if total_images > 0 else 0
            
            metrics = {
                'model': model_name,
                'model_path': model_path,
                'model_size_mb': round(get_model_size(model_path), 2),
                'avg_inference_time_ms': round(avg_inf, 2),
                'min_inference_time_ms': round(min_inf, 2),
                'max_inference_time_ms': round(max_inf, 2),
                'avg_fps': round(1000 / avg_inf, 2) if avg_inf > 0 else 0,
                'cpu_usage_percent': round(cpu_usage, 2),
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
    
    def print_comparison(self, original_metrics, pruned_metrics):
        print("\n" + "="*80)
        print("PRUNING COMPARISON RESULTS")
        print("="*80)
        
        print(f"\n{'Metric':<30} {'Original':<20} {'Pruned':<20} {'Change':<15}")
        print("-"*80)
        
        # Size comparison
        orig_size = original_metrics['model_size_mb']
        prun_size = pruned_metrics['model_size_mb']
        size_change = ((prun_size - orig_size) / orig_size * 100) if orig_size > 0 else 0
        print(f"{'Model Size (MB)':<30} {orig_size:<20.2f} {prun_size:<20.2f} {size_change:>+13.2f}%")
        
        # Inference time
        orig_time = original_metrics['avg_inference_time_ms']
        prun_time = pruned_metrics['avg_inference_time_ms']
        time_change = ((prun_time - orig_time) / orig_time * 100) if orig_time > 0 else 0
        print(f"{'Avg Inference Time (ms)':<30} {orig_time:<20.2f} {prun_time:<20.2f} {time_change:>+13.2f}%")
        
        # FPS
        orig_fps = original_metrics['avg_fps']
        prun_fps = pruned_metrics['avg_fps']
        fps_change = ((prun_fps - orig_fps) / orig_fps * 100) if orig_fps > 0 else 0
        print(f"{'Avg FPS':<30} {orig_fps:<20.2f} {prun_fps:<20.2f} {fps_change:>+13.2f}%")
        
        # Memory
        orig_mem = original_metrics['peak_memory_mb']
        prun_mem = pruned_metrics['peak_memory_mb']
        mem_change = ((prun_mem - orig_mem) / orig_mem * 100) if orig_mem > 0 else 0
        print(f"{'Peak Memory (MB)':<30} {orig_mem:<20.2f} {prun_mem:<20.2f} {mem_change:>+13.2f}%")
        
        # Accuracy metrics
        print("\n" + "-"*80)
        orig_acc = original_metrics['accuracy']
        prun_acc = pruned_metrics['accuracy']
        acc_change = ((prun_acc - orig_acc) / orig_acc * 100) if orig_acc > 0 else 0
        print(f"{'Accuracy (%)':<30} {orig_acc:<20.2f} {prun_acc:<20.2f} {acc_change:>+13.2f}%")
        
        orig_prec = original_metrics['precision']
        prun_prec = pruned_metrics['precision']
        prec_change = ((prun_prec - orig_prec) / orig_prec * 100) if orig_prec > 0 else 0
        print(f"{'Precision (%)':<30} {orig_prec:<20.2f} {prun_prec:<20.2f} {prec_change:>+13.2f}%")
        
        orig_rec = original_metrics['recall']
        prun_rec = pruned_metrics['recall']
        rec_change = ((prun_rec - orig_rec) / orig_rec * 100) if orig_rec > 0 else 0
        print(f"{'Recall (%)':<30} {orig_rec:<20.2f} {prun_rec:<20.2f} {rec_change:>+13.2f}%")
        
        orig_f1 = original_metrics['f1_score']
        prun_f1 = pruned_metrics['f1_score']
        f1_change = ((prun_f1 - orig_f1) / orig_f1 * 100) if orig_f1 > 0 else 0
        print(f"{'F1 Score (%)':<30} {orig_f1:<20.2f} {prun_f1:<20.2f} {f1_change:>+13.2f}%")
        
        print("="*80 + "\n")
    
    def run(self):
        models_dir = Path(self.paths['models_dir'])
        original_model = models_dir / f"{self.model_config['name']}.pt"
        
        if not original_model.exists():
            print(f"Error: Original model not found at {original_model}")
            return
        
        validation_images = self.collect_validation_images()
        if not validation_images:
            print("Warning: No validation images found")
            return
        
        print(f"Found {len(validation_images)} validation images\n")
        
        # Prune model
        pruned_model_path = self.prune_model(str(original_model))
        if not pruned_model_path:
            return
        
        # Benchmark both models
        print()
        original_metrics = self.benchmark_model(str(original_model), "Original", validation_images)
        pruned_metrics = self.benchmark_model(pruned_model_path, "Pruned", validation_images)
        
        if original_metrics and pruned_metrics:
            self.print_comparison(original_metrics, pruned_metrics)


if __name__ == "__main__":
    pruning = ModelPruning()
    pruning.run()
