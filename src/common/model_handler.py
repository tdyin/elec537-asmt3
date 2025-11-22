import os
import cv2
import numpy as np
from ultralytics import YOLO
from .utils import get_model_size, ensure_dir, nms

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


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
