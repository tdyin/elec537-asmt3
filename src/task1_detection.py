import time
import cv2
import numpy as np
from utils import load_config, get_cpu_usage, get_model_size, calculate_metrics
from utils import download_model_if_needed, ModelHandler


class ObjectDetector:
    """Object Detection using YOLO model with camera input"""
    def __init__(self, config_path="src/config.yaml"):
        self.config = load_config(config_path)
        self.camera_config = self.config['camera']
        self.model_config = self.config['model']
        self.paths = self.config['paths']
        self.task1_config = self.config['task1']
        self.monitoring = self.config['monitoring']
        
        model_path = download_model_if_needed(
            self.model_config['name'], self.paths['models_dir']
        )
        self.model = ModelHandler(model_path, "fp32")
        self.inference_times, self.cpu_usages = [], []
        self.frame_count = 0
        self.current_cpu_usage = 0.0 
        
    def initialize_camera(self):
        """Initialize camera for video capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            width, height = self.camera_config['resolution']
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_config['fps'])
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def process_frame(self, frame):
        """Process a single frame for object detection"""
        start_time = time.time()
        imgsz = self.model_config.get('imgsz', 320)
        
        results = self.model.model(
            frame, conf=self.model_config['confidence_threshold'],
            iou=self.model_config['iou_threshold'],
            device=self.model_config['device'], imgsz=imgsz,
            verbose=False, half=False, augment=False, max_det=100
        )
        
        inference_time = (time.time() - start_time) * 1000
        annotated_frame = results[0].plot()
        
        return annotated_frame, inference_time, results[0]
    
    def display_metrics(self, frame, inference_time, cpu_usage):
        """Overlay performance metrics on the frame"""
        avg_inference = np.mean(self.inference_times[-30:]) if self.inference_times else 0
        fps = 1000 / avg_inference if avg_inference > 0 else 0
        
        frame_height, frame_width = frame.shape[:2]
        
        font_scale = max(0.4, min(0.8, frame_height / 800))
        thickness = max(1, int(font_scale * 4))
        line_spacing = int(frame_height * 0.08)  
        
        x_offset = int(frame_width * 0.015) 
        y_offset = int(frame_height * 0.05) 
        
        metrics = [
            f"Inference: {inference_time:.1f}ms",
            f"FPS: {fps:.1f}",
            f"CPU: {cpu_usage:.1f}%",
            f"Frame: {self.frame_count}"
        ]
        
        for metric in metrics:
            cv2.putText(frame, metric, (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            y_offset += line_spacing
        
        return frame
    
    def run(self):
        """Main loop to run object detection"""
        if not self.initialize_camera():
            return
        
        print("\nStarting object detection (Press 'q' to quit)\n")
        frame_skip, skip_counter = 0, 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                self.frame_count += 1
                skip_counter += 1
                
                if skip_counter <= frame_skip:
                    continue
                skip_counter = 0
                
                annotated_frame, inference_time, results = self.process_frame(frame)
                
                if self.frame_count % 5 == 0:
                    self.current_cpu_usage = get_cpu_usage()
                    if self.frame_count > self.monitoring['warmup_frames']:
                        self.cpu_usages.append(self.current_cpu_usage)
                
                if self.frame_count > self.monitoring['warmup_frames']:
                    self.inference_times.append(inference_time)
                
                display_frame = self.display_metrics(annotated_frame, inference_time, self.current_cpu_usage)
                
                if self.task1_config['display_preview']:
                    cv2.imshow('Object Detection', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
            self.print_summary()
    
    def cleanup(self):
        """Release resources and close windows"""
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
    
    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        if self.inference_times:
            min_inf, max_inf, avg_inf = calculate_metrics(self.inference_times)
            print(f"Average Inference Time: {avg_inf:.2f} ms")
            print(f"Min Inference Time: {min_inf:.2f} ms")
            print(f"Max Inference Time: {max_inf:.2f} ms")
            print(f"Average FPS: {1000/avg_inf:.2f}")
        
        if self.cpu_usages:
            min_cpu, max_cpu, avg_cpu = calculate_metrics(self.cpu_usages)
            print(f"Average CPU Utilization: {avg_cpu:.1f}%")
            print(f"Min CPU Utilization: {min_cpu:.1f}%")
            print(f"Max CPU Utilization: {max_cpu:.1f}%")
        
        model_size = self.model.get_size()
        print(f"Model Size: {model_size:.2f} MB")
        print(f"Total Frames Processed: {self.frame_count}")
        print("="*60)


if __name__ == "__main__":
    detector = ObjectDetector()
    detector.run()
