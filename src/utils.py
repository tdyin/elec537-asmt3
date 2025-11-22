import os
import yaml
import psutil
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    
'''
Utility functions 
'''  
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model_size(model_path):
    """Get model file size in MB"""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 * 1024)
    return 0


def get_cpu_usage():
    """Get current CPU usage percentage"""
    return psutil.cpu_percent(interval=0.1)


def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def calculate_metrics(values):
    """Calculate min, max, and mean of a list of values"""
    if not values:
        return 0, 0, 0
    return np.min(values), np.max(values), np.mean(values)


def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression for bounding boxes"""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    
    # Coordinates of bounding boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    # Perform Non-Maximum Suppression
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

'''
Model Handler
'''
class ModelHandler:
    """Unified handler for PyTorch and ONNX models"""
    
    def __init__(self, model_path, model_type="fp32"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.ort_session = None
        self.device = ort.get_available_providers() if ONNX_AVAILABLE else None
        
        self._load_model()
    
    def _load_model(self):
        """Load model based on type"""
        if self.model_path.endswith('.pt'):
            self.model = YOLO(self.model_path, task='detect')
        elif self.model_path.endswith('.onnx') and ONNX_AVAILABLE:
            providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in self.device else ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session = ort.InferenceSession(
                self.model_path, sess_options=sess_options,
                providers=providers
            )
    
    def predict(self, img, conf_threshold=0.4, verbose=False):
        """Run inference on image"""
        if self.model:
            return self.model(img, conf=conf_threshold, verbose=verbose)
        elif self.ort_session:
            input_tensor = self._preprocess_onnx(img)
            outputs = self.ort_session.run(
                None, {self.ort_session.get_inputs()[0].name: input_tensor}
            )
            return self._postprocess_onnx(outputs, conf_threshold)
        return None
    
    def _preprocess_onnx(self, img):
        """Preprocess image for ONNX"""
        img_resized = cv2.resize(img, (640, 640)) # Assuming model input size is 640x640
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        img_normalized = img_rgb.astype(np.float32) / 255.0 # Normalize to [0, 1]
        img_chw = np.transpose(img_normalized, (2, 0, 1)) # HWC to CHW
        img_batch = np.expand_dims(img_chw, axis=0) # Add batch dimension
        
        # Convert to fp16 if model expects fp16 input
        if self.ort_session and self.ort_session.get_inputs()[0].type == 'tensor(float16)':
            img_batch = img_batch.astype(np.float16)
        
        return img_batch
    
    def _postprocess_onnx(self, outputs, conf_threshold):
        """Extract detections from ONNX output"""
        predictions = outputs[0]

        # Adjust shape if necessary
        if len(predictions.shape) == 3 and predictions.shape[1] < predictions.shape[2]:
            predictions = np.transpose(predictions, (0, 2, 1))
            
        # Assuming predictions shape is (1, N, 4 + num_classes)
        predictions = predictions[0]
        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]
        
        # Get max class scores and corresponding class IDs
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Filter by confidence threshold
        mask = max_scores > conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = class_ids[mask]
        
        # Apply NMS
        if len(filtered_boxes) > 0:
            x_c, y_c, w, h = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3]
            boxes_corner = np.stack([x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2], axis=1)
            keep = nms(boxes_corner, filtered_scores, iou_threshold=0.45)
            return filtered_classes[keep], filtered_scores[keep]
        
        return np.array([]), np.array([])
    
    def get_size(self):
        """Get model size in MB"""
        return get_model_size(self.model_path)


def download_model_if_needed(model_name, models_dir):
    """Download model if not present locally"""
    ensure_dir(models_dir)
    model_path = f"{models_dir}/{model_name}.pt"
    
    if not os.path.exists(model_path):
        print("Model not found locally. Downloading...")
        model = YOLO(model_name)
        model.save(model_path)
    
    return model_path
