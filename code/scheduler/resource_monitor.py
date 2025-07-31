"""
Resource Monitor for LLM Evaluation
Real-time GPU/CPU/memory monitoring and OOM prediction
"""
import time
import threading
import psutil
import GPUtil
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
from pathlib import Path
import json

# Logging setup
logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Resource usage snapshot"""
    timestamp: float
    
    # GPU information
    gpu_memory_used: float  # GB
    gpu_memory_total: float  # GB
    gpu_memory_percent: float  # %
    gpu_utilization: float  # %
    gpu_temperature: float  # °C
    
    # CPU information
    cpu_percent: float  # %
    cpu_per_core: List[float]  # Per-core usage
    
    # RAM information
    ram_used: float  # GB
    ram_total: float  # GB
    ram_percent: float  # %
    ram_available: float  # GB
    
    # Process-specific information (optional)
    process_memory: Optional[float] = None  # GB
    process_cpu: Optional[float] = None  # %


class ResourceMonitor:
    """Real-time resource monitoring class"""
    
    def __init__(self, 
                 monitoring_interval: float = 1.0,
                 history_size: int = 300,  # 5 minutes of records (1 second interval)
                 gpu_index: int = 0):
        """
        Args:
            monitoring_interval: Monitoring interval (seconds)
            history_size: History size
            gpu_index: GPU index to monitor
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.gpu_index = gpu_index
        
        # History storage (deque with automatic size limit)
        self.history = deque(maxlen=history_size)
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
        # Thresholds for OOM prediction
        self.oom_threshold_percent = 90  # GPU memory 90% or higher
        self.oom_warning_percent = 80   # GPU memory 80% or higher
        
        # Callback functions
        self.oom_warning_callback: Optional[Callable] = None
        self.oom_critical_callback: Optional[Callable] = None
        
        logger.info(f"ResourceMonitor initialized: GPU {gpu_index}, interval {monitoring_interval}s")
    
    def start_monitoring(self):
        """Start monitoring"""
        if self._monitoring:
            logger.warning("Monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Monitoring loop (runs in separate thread)"""
        while self._monitoring:
            try:
                snapshot = self._take_snapshot()
                self.history.append(snapshot)
                
                # OOM check
                self._check_oom_risk(snapshot)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Current resource usage snapshot"""
        # GPU information
        gpu_memory_used = 0
        gpu_memory_total = 0
        gpu_memory_percent = 0
        gpu_utilization = 0
        gpu_temperature = 0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus and self.gpu_index < len(gpus):
                gpu = gpus[self.gpu_index]
                gpu_memory_used = gpu.memoryUsed / 1024  # MB to GB
                gpu_memory_total = gpu.memoryTotal / 1024  # MB to GB
                gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0
                gpu_utilization = gpu.load * 100
                gpu_temperature = gpu.temperature
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
        
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # RAM information
        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024**3)  # bytes to GB
        ram_total = ram.total / (1024**3)
        ram_percent = ram.percent
        ram_available = ram.available / (1024**3)
        
        # Current process information (optional)
        process_memory = None
        process_cpu = None
        try:
            current_process = psutil.Process()
            process_memory = current_process.memory_info().rss / (1024**3)  # bytes to GB
            process_cpu = current_process.cpu_percent(interval=0.1)
        except:
            pass
        
        return ResourceSnapshot(
            timestamp=time.time(),
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_memory_percent=gpu_memory_percent,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature,
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            ram_used=ram_used,
            ram_total=ram_total,
            ram_percent=ram_percent,
            ram_available=ram_available,
            process_memory=process_memory,
            process_cpu=process_cpu
        )
    
    def _check_oom_risk(self, snapshot: ResourceSnapshot):
        """Check OOM risk and call callbacks"""
        if snapshot.gpu_memory_percent >= self.oom_threshold_percent:
            if self.oom_critical_callback:
                self.oom_critical_callback(snapshot)
            logger.critical(f"CRITICAL OOM RISK: GPU memory at {snapshot.gpu_memory_percent:.1f}%")
        
        elif snapshot.gpu_memory_percent >= self.oom_warning_percent:
            if self.oom_warning_callback:
                self.oom_warning_callback(snapshot)
            logger.warning(f"OOM WARNING: GPU memory at {snapshot.gpu_memory_percent:.1f}%")
    
    def get_current_snapshot(self) -> Optional[ResourceSnapshot]:
        """Get current resource usage immediately"""
        try:
            return self._take_snapshot()
        except Exception as e:
            logger.error(f"Failed to take snapshot: {e}")
            return None
    
    def get_history(self, seconds: Optional[int] = None) -> List[ResourceSnapshot]:
        """Get history
        
        Args:
            seconds: Return only data from last N seconds (None for all)
        """
        if seconds is None:
            return list(self.history)
        
        current_time = time.time()
        cutoff_time = current_time - seconds
        return [s for s in self.history if s.timestamp >= cutoff_time]
    
    def get_statistics(self, seconds: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Calculate statistics
        
        Returns:
            Mean, max, min, standard deviation for each metric
        """
        history = self.get_history(seconds)
        if not history:
            return {}
        
        # Collect values for each metric
        metrics = {
            'gpu_memory_used': [s.gpu_memory_used for s in history],
            'gpu_memory_percent': [s.gpu_memory_percent for s in history],
            'gpu_utilization': [s.gpu_utilization for s in history],
            'gpu_temperature': [s.gpu_temperature for s in history],
            'cpu_percent': [s.cpu_percent for s in history],
            'ram_used': [s.ram_used for s in history],
            'ram_percent': [s.ram_percent for s in history],
        }
        
        # Calculate statistics
        stats = {}
        for metric_name, values in metrics.items():
            if values:
                stats[metric_name] = {
                    'current': values[-1] if values else 0,
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values),
                }
        
        return stats
    
    def predict_oom_probability(self, lookahead_seconds: int = 60) -> float:
        """Predict OOM occurrence probability
        
        Args:
            lookahead_seconds: Prediction time (seconds)
            
        Returns:
            OOM occurrence probability (0.0 ~ 1.0)
        """
        history = self.get_history(seconds=60)  # Last 1 minute of data
        if len(history) < 10:
            return 0.0  # Insufficient data
        
        # Analyze GPU memory usage trend
        memory_percents = [s.gpu_memory_percent for s in history]
        times = [s.timestamp for s in history]
        
        # Calculate trend using linear regression
        if len(memory_percents) >= 2:
            # Normalize time (first time point as 0)
            times_normalized = np.array(times) - times[0]
            
            # Linear regression
            coeffs = np.polyfit(times_normalized, memory_percents, 1)
            slope = coeffs[0]  # Rate of increase per second
            
            # Current memory usage
            current_percent = memory_percents[-1]
            
            # Predicted usage after lookahead_seconds
            predicted_percent = current_percent + (slope * lookahead_seconds)
            
            # Calculate OOM probability (using sigmoid function)
            # Probability increases rapidly above 90%
            if predicted_percent >= 100:
                return 1.0
            elif predicted_percent >= 90:
                return 0.8 + (predicted_percent - 90) * 0.02
            elif predicted_percent >= 80:
                return 0.5 + (predicted_percent - 80) * 0.03
            else:
                return predicted_percent / 200  # Max 0.4
        
        return 0.0
    
    def get_memory_growth_rate(self) -> float:
        """Calculate memory growth rate (GB/second)"""
        history = self.get_history(seconds=30)  # Last 30 seconds
        if len(history) < 2:
            return 0.0
        
        memory_values = [s.gpu_memory_used for s in history]
        time_diff = history[-1].timestamp - history[0].timestamp
        
        if time_diff > 0:
            memory_diff = memory_values[-1] - memory_values[0]
            return memory_diff / time_diff
        
        return 0.0
    
    def estimate_memory_for_batch_size(self, current_batch_size: int, 
                                     new_batch_size: int) -> float:
        """Estimate memory usage for different batch size
        
        Args:
            current_batch_size: Current batch size
            new_batch_size: New batch size
            
        Returns:
            Estimated GPU memory usage (GB)
        """
        current_snapshot = self.get_current_snapshot()
        if not current_snapshot:
            return 0.0
        
        # Simple linear estimation (actual behavior may be more complex)
        current_memory = current_snapshot.gpu_memory_used
        
        # Assume proportional to batch size
        if current_batch_size > 0:
            estimated_memory = current_memory * (new_batch_size / current_batch_size)
            return min(estimated_memory, current_snapshot.gpu_memory_total)
        
        return current_memory
    
    def recommend_batch_size(self, target_memory_percent: float = 70) -> int:
        """Recommend safe batch size
        
        Args:
            target_memory_percent: Target memory usage (%)
            
        Returns:
            Recommended batch size
        """
        current_snapshot = self.get_current_snapshot()
        if not current_snapshot:
            return 1
        
        # If current memory usage is already high, use batch size 1
        if current_snapshot.gpu_memory_percent >= target_memory_percent:
            return 1
        
        # Calculate available memory
        total_memory = current_snapshot.gpu_memory_total
        target_memory = total_memory * (target_memory_percent / 100)
        current_memory = current_snapshot.gpu_memory_used
        available_memory = target_memory - current_memory
        
        # Simple estimation: assume current batch size is 1, linear scaling
        # Actual behavior depends on model and task
        if available_memory > 0 and current_memory > 0:
            scale_factor = target_memory / current_memory
            return max(1, int(scale_factor))
        
        return 1
    
    def save_history(self, filepath: Path):
        """Save history to file"""
        # Add timestamp to filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = filepath.stem
        suffix = filepath.suffix
        filepath = filepath.parent / f"{stem}_{timestamp}{suffix}"
        
        data = {
            'metadata': {
                'gpu_index': self.gpu_index,
                'monitoring_interval': self.monitoring_interval,
                'saved_at': datetime.now().isoformat(),
            },
            'history': [
                {
                    'timestamp': s.timestamp,
                    'gpu_memory_used': s.gpu_memory_used,
                    'gpu_memory_percent': s.gpu_memory_percent,
                    'gpu_utilization': s.gpu_utilization,
                    'cpu_percent': s.cpu_percent,
                    'ram_percent': s.ram_percent,
                }
                for s in self.history
            ]
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Resource history saved to {filepath}")
    
    def __enter__(self):
        """Context manager support"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop_monitoring()