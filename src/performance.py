"""
Performance monitoring utilities for tracking system metrics.
"""

import psutil
import time
import threading
from collections import deque


class PerformanceMonitor:
    """Monitors CPU, memory, and inference performance."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.cpu_readings = deque(maxlen=window_size)
        self.memory_readings = deque(maxlen=window_size)
        self.monitoring = False
        self.monitor_thread = None
        
    def add_inference_time(self, time_ms):
        """Record an inference time."""
        self.inference_times.append(time_ms)
    
    def start_monitoring(self, interval=1.0):
        """Start background monitoring of system metrics."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval):
        """Background loop for monitoring system metrics."""
        while self.monitoring:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory().percent
            
            self.cpu_readings.append(cpu)
            self.memory_readings.append(memory)
            
            time.sleep(interval)
    
    def get_current_stats(self):
        """Get current performance statistics."""
        stats = {
            'avg_inference_ms': 0,
            'min_inference_ms': 0,
            'max_inference_ms': 0,
            'avg_fps': 0,
            'avg_cpu_percent': 0,
            'avg_memory_percent': 0,
            'sample_count': len(self.inference_times)
        }
        
        if len(self.inference_times) > 0:
            times = list(self.inference_times)
            stats['avg_inference_ms'] = sum(times) / len(times)
            stats['min_inference_ms'] = min(times)
            stats['max_inference_ms'] = max(times)
            stats['avg_fps'] = 1000 / stats['avg_inference_ms']
        
        if len(self.cpu_readings) > 0:
            stats['avg_cpu_percent'] = sum(self.cpu_readings) / len(self.cpu_readings)
        
        if len(self.memory_readings) > 0:
            stats['avg_memory_percent'] = sum(self.memory_readings) / len(self.memory_readings)
        
        return stats
    
    def print_summary(self):
        """Print performance summary."""
        stats = self.get_current_stats()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Average Inference Time: {stats['avg_inference_ms']:.2f} ms")
        print(f"Min Inference Time: {stats['min_inference_ms']:.2f} ms")
        print(f"Max Inference Time: {stats['max_inference_ms']:.2f} ms")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Average CPU Usage: {stats['avg_cpu_percent']:.1f}%")
        print(f"Average Memory Usage: {stats['avg_memory_percent']:.1f}%")
        print(f"Samples Collected: {stats['sample_count']}")
        print("="*60)


def get_system_info():
    """Get detailed system information."""
    info = {
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(logical=True),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
    }
    return info


def print_system_info():
    """Print system information."""
    info = get_system_info()
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"CPU Cores: {info['cpu_count']} ({info['cpu_threads']} threads)")
    print(f"Total Memory: {info['memory_total_gb']:.2f} GB")
    print(f"Available Memory: {info['memory_available_gb']:.2f} GB")
    print("="*60)
