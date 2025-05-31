import time
import threading
import pandas as pd
import torch
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    nvmlShutdown
)

class GPUMonitor:
    def __init__(self, log_file: str, gpu_id=0, save_interval=300):
        """
        Initialize GPU Monitor
        
        Args:
            gpu_id (int): GPU device ID to monitor
            log_file (str): Path to save the GPU monitoring logs
            save_interval (int): Interval in seconds to save logs
        """
        self.gpu_id = gpu_id
        self.log_file = log_file
        self.save_interval = save_interval
        self.gpu_stats = []
        self.training_running = False
        self.start_time = None
        self.monitor_thread = None
        
        # Initialize NVIDIA Management Library
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_id)
    
    def start_monitoring(self):
        """Start GPU monitoring in a separate thread"""
        self.training_running = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_gpu)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop GPU monitoring and save final logs"""
        self.training_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.save_logs()
        self._print_summary()
        nvmlShutdown()
    
    def _monitor_gpu(self):
        """Monitor GPU usage in real-time"""
        last_save_time = time.time()
        
        while self.training_running:
            utilization = nvmlDeviceGetUtilizationRates(self.handle).gpu
            memory_used = nvmlDeviceGetMemoryInfo(self.handle).used / 1024**3  # Convert to GB
            current_time = time.time() - self.start_time
            self.gpu_stats.append((current_time, utilization, memory_used))

            # Save logs periodically
            if time.time() - last_save_time >= self.save_interval:
                self.save_logs()
                last_save_time = time.time()

            time.sleep(5)  # Log every 5 seconds
    
    def save_logs(self):
        """Save GPU usage logs to CSV file"""
        df = pd.DataFrame(
            self.gpu_stats,
            columns=["Time (s)", "GPU Utilization (%)", "Memory Used (GB)"]
        )
        df.to_csv(self.log_file, index=False)
        print(f"Logs saved to {self.log_file}")
    
    def _print_summary(self):
        """Print training summary including total time and peak memory usage"""
        if self.start_time:
            total_time = time.time() - self.start_time
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
            
            print(f"Total Training Time: {total_time:.2f} seconds")
            print(f"Peak GPU Memory Usage: {peak_memory:.2f} GB")