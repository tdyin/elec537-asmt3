"""
Utility functions for video recording and frame capture.
"""

import cv2
import time
from datetime import datetime
from pathlib import Path


class VideoRecorder:
    """Records video from camera for demonstration purposes."""
    
    def __init__(self, output_dir="./results/videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.is_recording = False
        
    def start_recording(self, fps=30, resolution=(640, 480)):
        """Start video recording."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"detection_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(filename), fourcc, fps, resolution)
        self.is_recording = True
        
        print(f"Started recording: {filename}")
        return filename
    
    def write_frame(self, frame):
        """Write a frame to video."""
        if self.is_recording and self.writer is not None:
            self.writer.write(frame)
    
    def stop_recording(self):
        """Stop video recording."""
        if self.writer is not None:
            self.writer.release()
            self.is_recording = False
            print("Recording stopped")


class FrameCapture:
    """Captures individual frames with metadata."""
    
    def __init__(self, output_dir="./results/captured_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_frame(self, frame, metadata=None):
        """Save frame with optional metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.output_dir / f"frame_{timestamp}.jpg"
        
        cv2.imwrite(str(filename), frame)
        
        # Save metadata if provided
        if metadata:
            meta_filename = filename.with_suffix('.txt')
            with open(meta_filename, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
        
        return filename


def draw_fps(frame, fps, position=(10, 30)):
    """Draw FPS counter on frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", position, 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def calculate_fps(inference_times, window_size=30):
    """Calculate FPS from recent inference times."""
    if len(inference_times) == 0:
        return 0
    
    recent_times = inference_times[-window_size:]
    avg_time = sum(recent_times) / len(recent_times)
    fps = 1000 / avg_time if avg_time > 0 else 0
    return fps
