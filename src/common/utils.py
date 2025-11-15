"""Shared utilities for model operations and benchmarking"""
import os
import yaml
import psutil
import numpy as np
from pathlib import Path


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
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
