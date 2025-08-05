"""
Intelligent Scheduler for LLM Evaluation
Optimal execution order and resource allocation decisions based on learned data
FIXED: Added quantization-aware memory estimation
FIXED: Integrated with DynamicModelClassifier for consistent model analysis
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import json

# FIXED: Use absolute import paths instead of relative imports
try:
    from code.scheduler.performance_tracker import PerformanceTracker
    from code.scheduler.resource_monitor import ResourceMonitor
except ImportError:
    # Fallback to relative imports for direct execution
    from .performance_tracker import PerformanceTracker
    from .resource_monitor import ResourceMonitor

# Import DynamicModelClassifier for consistent model analysis
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from evaluation_lm import DynamicModelClassifier
except ImportError:
    # Fallback: create a simplified version if import fails
    class DynamicModelClassifier:
        def __init__(self):
            self.quantization_memory_factors = {
                '4bit': 0.25, '8bit': 0.5, 'fp16': 0.6, 'fp32': 1.0
            }
        
        def extract_model_size(self, model_config: Dict) -> float:
            model_id = model_config.get("id", "")
            size_patterns = {
                '32b': 32.0, '70b': 70.0, '13b': 13.0, '7b': 7.0,
                '30b': 30.0, '21.4b': 21.4, '12b': 12.0, '8b': 8.0,
                '4b': 4.0, '3b': 3.0, '1.5b': 1.5, '0.5b': 0.5
            }
            model_id_lower = model_id.lower()
            for pattern, size in size_patterns.items():
                if pattern in model_id_lower:
                    return size
            return 7.0
        
        def classify_model(self, model_config: Dict) -> Tuple[str, float, Dict]:
            size_b = self.extract_model_size(model_config)
            if size_b < 3.0:
                category = 'small'
            elif size_b < 10.0:
                category = 'medium'
            else:
                category = 'large'
            
            config = {
                'use_hflm': category == 'large',
                'load_in_8bit': category == 'large',
                'device_map': 'auto' if category in ['large', 'medium'] else None,
                'default_batch_size': 1 if category == 'large' else (2 if category == 'medium' else 4)
            }
            return category, size_b, config

# Logging setup
logger = logging.getLogger(__name__)


@dataclass
class TaskPriority:
    """Task priority information"""
    model_id: str
    task_name: str
    priority_score: float
    estimated_time: float
    estimated_memory: float
    success_probability: float
    
    # Scheduling decisions
    suggested_gpu: int
    suggested_batch_size: int
    suggested_num_fewshot: int
    
    # Additional metadata
    rationale: str


class IntelligentScheduler:
    """Intelligent scheduler class with quantization-aware memory estimation and DynamicModelClassifier integration"""
    
    def __init__(self, 
                 performance_tracker: PerformanceTracker,
                 resource_monitor: ResourceMonitor,
                 num_gpus: int = 1):
        """
        Args:
            performance_tracker: Performance tracker instance
            resource_monitor: Resource monitor instance
            num_gpus: Number of available GPUs
        """
        self.performance_tracker = performance_tracker
        self.resource_monitor = resource_monitor
        self.num_gpus = num_gpus
        
        # Initialize DynamicModelClassifier for consistent model analysis
        self.model_classifier = DynamicModelClassifier()
        
        # Scheduling policies
        self.memory_safety_margin = 0.15  # 15% margin
        self.prefer_small_models_first = True  # Prioritize small models
        self.balance_gpu_load = True  # Balance GPU load distribution
        
        logger.info(f"IntelligentScheduler initialized with {num_gpus} GPUs (integrated with DynamicModelClassifier)")
    
    def _extract_model_size(self, model_id: str) -> str:
        """Extract model size from model ID (deprecated - use DynamicModelClassifier instead)"""
        logger.warning("Using deprecated _extract_model_size, consider using DynamicModelClassifier")
        return self._extract_model_size_legacy(model_id)
    
    def _estimate_time_by_size(self, model_size: str, task: str) -> float:
        """Estimate execution time by model size and task (deprecated - use dynamic estimation)"""
        logger.warning("Using deprecated _estimate_time_by_size, consider using dynamic estimation")
        # Create a mock model dict for the new method
        mock_model = {"id": f"mock/{model_size.lower()}-model", "name": f"{model_size} Model"}
        return self._estimate_time_using_dynamic_classifier(mock_model, task)
    
    def _estimate_memory_by_size_quantization_aware(self, model: Dict, task: str) -> float:
        """Estimate memory usage by model size and task with quantization awareness (deprecated)"""
        logger.warning("Using deprecated _estimate_memory_by_size_quantization_aware, using dynamic estimation instead")
        return self._estimate_memory_using_dynamic_classifier(model, task)
    
    def _extract_model_size_numeric(self, model_id: str) -> float:
        """Extract numeric model size in billions (now using DynamicModelClassifier)"""
        try:
            mock_model = {"id": model_id, "name": model_id.split("/")[-1]}
            return self.model_classifier.extract_model_size(mock_model)
        except:
            # Fallback to legacy method
            model_id_lower = model_id.lower()
            size_patterns = {
                '0.5b': 0.5, '1.5b': 1.5, '2.1b': 2.1, '2.4b': 2.4,
                '3b': 3.0, '4b': 4.0, '7b': 7.0, '7.8b': 7.8, '8b': 8.0,
                '12b': 12.0, '13b': 13.0, '21.4b': 21.4, '30b': 30.0,
                '32b': 32.0, '70b': 70.0
            }
            
            for pattern, size in size_patterns.items():
                if pattern in model_id_lower:
                    return size
            
            return 7.0  # Default
    
    def _apply_quantization_adjustment_old(self, raw_memory: float, model: Dict) -> float:
        """Old quantization adjustment method (deprecated)"""
        quantization_type = self._extract_quantization_type(model)
        return self._apply_quantization_adjustment(raw_memory, quantization_type)
    
    def create_optimal_schedule(self, 
                              models: List[Dict], 
                              tasks: List[str]) -> List[TaskPriority]:
        """Create optimal execution schedule
        
        Args:
            models: List of model configurations
            tasks: List of task names
            
        Returns:
            Priority-sorted list of TaskPriority
        """
        schedule = []
        
        # Calculate priority for all model-task combinations
        for model in models:
            for task in tasks:
                priority = self._calculate_priority(model, task)
                if priority:
                    schedule.append(priority)
        
        # Sort by priority
        schedule.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Optimize GPU assignment
        self._optimize_gpu_assignment(schedule)
        
        logger.info(f"Created optimal schedule with {len(schedule)} tasks")
        return schedule
    
    def _calculate_priority(self, model: Dict, task: str) -> Optional[TaskPriority]:
        """Calculate priority for individual task using DynamicModelClassifier"""
        model_id = model.get("id", "")
        model_name = model.get("name", model_id.split("/")[-1])
        
        # Performance prediction
        prediction = self.performance_tracker.predict_execution(model_id, task)
        
        # Use dynamic estimation with DynamicModelClassifier
        if prediction['predicted_time'] is None:
            estimated_time = self._estimate_time_using_dynamic_classifier(model, task)
            estimated_memory = self._estimate_memory_using_dynamic_classifier(model, task)
            success_probability = 0.8  # Default value
        else:
            estimated_time = prediction['predicted_time']
            # Apply quantization adjustment to predicted memory if needed
            raw_memory = prediction['predicted_memory'] or 10.0
            quantization_type = self._extract_quantization_type(model)
            estimated_memory = self._apply_quantization_adjustment(raw_memory, quantization_type)
            success_probability = prediction['success_rate'] or 0.8
        
        # Calculate OOM risk with quantization consideration
        oom_risk = prediction.get('oom_rate', 0.0)
        if oom_risk is None:
            oom_risk = 0.0
        
        # Adjust OOM risk based on quantization
        quantization_type = self._extract_quantization_type(model)
        if quantization_type in ['4bit', '8bit']:
            oom_risk *= (0.7 if quantization_type == '4bit' else 0.8)
        
        # Get model classification for priority calculation
        try:
            category, size_b, opt_config = self.model_classifier.classify_model(model)
            model_size_str = f"{size_b:.1f}B"
        except:
            model_size_str = self._extract_model_size_legacy(model_id)
        
        # Calculate priority score
        priority_score = self._compute_priority_score(
            estimated_time=estimated_time,
            estimated_memory=estimated_memory,
            success_probability=success_probability,
            oom_risk=oom_risk,
            model_size=model_size_str
        )
        
        # Determine batch size and few-shot using dynamic estimation
        suggested_batch_size = self._suggest_batch_size(estimated_memory)
        suggested_num_fewshot = self._suggest_num_fewshot(task, estimated_memory)
        
        # Generate rationale with quantization info
        rationale = self._generate_rationale(
            model_name, task, estimated_time, estimated_memory, 
            success_probability, oom_risk, quantization_type
        )
        
        return TaskPriority(
            model_id=model_id,
            task_name=task,
            priority_score=priority_score,
            estimated_time=estimated_time,
            estimated_memory=estimated_memory,
            success_probability=success_probability,
            suggested_gpu=0,  # Optimized later
            suggested_batch_size=suggested_batch_size,
            suggested_num_fewshot=suggested_num_fewshot,
            rationale=rationale
        )
    
    def _extract_quantization_type(self, model: Dict) -> str:
        """Extract quantization type from model configuration"""
        model_id = model.get("id", "").lower()
        model_name = model.get("name", "").lower()
        
        if any(q in model_id for q in ['4bit', '4-bit', 'bnb-4bit']):
            return '4bit'
        elif any(q in model_id for q in ['8bit', '8-bit', 'int8']):
            return '8bit'
        elif any(q in model_id for q in ['fp16', 'half']):
            return 'fp16'
        elif any(q in model_id for q in ['fp32', 'float32']):
            return 'fp32'
        else:
            # Infer quantization based on model size (large models likely use quantization)
            model_size_numeric = self._extract_model_size_numeric(model_id)
            if model_size_numeric >= 30:
                return '8bit'  # Large models likely use 8bit quantization
            elif model_size_numeric >= 10:
                return 'fp16'
            else:
                return 'fp16'
    
    def _extract_model_size_numeric(self, model_id: str) -> float:
        """Extract numeric model size in billions"""
        model_id_lower = model_id.lower()
        
        size_patterns = {
            '0.5b': 0.5, '0-5b': 0.5, '500m': 0.5,
            '1.5b': 1.5, '1-5b': 1.5,
            '2.1b': 2.1, '2-1b': 2.1, '2.4b': 2.4, '2-4b': 2.4,
            '3b': 3.0, '4b': 4.0, '7b': 7.0, '7.8b': 7.8, '7-8b': 7.8,
            '8b': 8.0, '12b': 12.0, '13b': 13.0, '21.4b': 21.4, '21-4b': 21.4,
            '30b': 30.0, '32b': 32.0, '70b': 70.0
        }
        
        for pattern, size in size_patterns.items():
            if pattern in model_id_lower:
                return size
        
        return 7.0  # Default
    
    def _apply_quantization_adjustment(self, raw_memory: float, model: Dict) -> float:
        """Apply quantization adjustment to memory prediction"""
        quantization_type = self._extract_quantization_type(model)
        adjustment_factor = self.quantization_memory_factors.get(quantization_type, 1.0)
        adjusted_memory = raw_memory * adjustment_factor
        
        # Add some safety margin for quantization overhead
        if quantization_type in ['4bit', '8bit']:
            overhead_factor = 1.1  # 10% overhead for quantization
            adjusted_memory *= overhead_factor
        
        logger.debug(f"Memory adjustment: {raw_memory:.1f}GB -> {adjusted_memory:.1f}GB "
                    f"(quantization: {quantization_type}, factor: {adjustment_factor:.2f})")
        
        return max(adjusted_memory, 1.0)  # Minimum 1GB
    
    def _estimate_memory_by_size_quantization_aware(self, model: Dict, task: str) -> float:
        """Estimate memory usage by model size and task with quantization awareness (GB)"""
        model_id = model.get("id", "")
        model_size = self._extract_model_size(model_id)
        
        # Base memory (GB) - these are for full precision
        base_memory = {
            "0.5B": 2,
            "1.5B": 4,
            "2.1B": 6,
            "2.4B": 7,
            "3B": 8,
            "4B": 12,
            "7B": 20,
            "7.8B": 22,
            "8B": 24,
            "12B": 35,
            "13B": 38,
            "21.4B": 55,
            "30B": 75,
            "32B": 80,  # Base full precision memory
            "70B": 140,
        }
        
        raw_memory = 20  # Default 20GB
        for size, mem in base_memory.items():
            if size in model_size.upper():
                raw_memory = mem
                break
        
        # Task-specific adjustment (longer sequences need more memory)
        if any(t in task.lower() for t in ["mmlu_pro", "bbh", "humaneval"]):
            raw_memory *= 1.2
        
        # Apply quantization adjustment
        adjusted_memory = self._apply_quantization_adjustment(raw_memory, model)
        
        return adjusted_memory
    
    def _compute_priority_score(self, 
                              estimated_time: float,
                              estimated_memory: float,
                              success_probability: float,
                              oom_risk: float,
                              model_size: str) -> float:
        """Calculate priority score
        
        Higher score = Higher priority
        """
        # Base score (success probability)
        score = success_probability * 100
        
        # Time efficiency (shorter is better)
        if estimated_time > 0:
            time_efficiency = 1 / (1 + np.log1p(estimated_time / 3600))  # 1 hour baseline
            score *= time_efficiency
        
        # Memory efficiency
        if estimated_memory > 0:
            memory_efficiency = 1 / (1 + np.log1p(estimated_memory / 40))  # 40GB baseline
            score *= memory_efficiency
        
        # OOM risk penalty
        score *= (1 - oom_risk)
        
        # Model size bonus/penalty
        if self.prefer_small_models_first:
            size_bonus = self._get_size_bonus(model_size)
            score *= size_bonus
        
        return score
    
    def _get_size_bonus(self, model_size: str) -> float:
        """Bonus/penalty based on model size (now using dynamic size detection)"""
        try:
            # Extract numeric size from string (e.g., "32.0B" -> 32.0)
            size_numeric = float(model_size.replace('B', '').replace('M', '').replace('K', ''))
            
            # Convert to billions if needed
            if 'M' in model_size:
                size_numeric /= 1000.0
            elif 'K' in model_size:
                size_numeric /= 1000000.0
            
            # Dynamic bonus calculation based on size
            if size_numeric <= 1.0:
                return 1.5
            elif size_numeric <= 3.0:
                return 1.3
            elif size_numeric <= 7.0:
                return 1.0
            elif size_numeric <= 13.0:
                return 0.8
            elif size_numeric <= 30.0:
                return 0.6
            else:
                return 0.5
                
        except:
            # Fallback to legacy mapping
            size_map = {
                "0.5B": 1.5, "1.5B": 1.4, "2.1B": 1.3, "2.4B": 1.3,
                "3B": 1.2, "4B": 1.1, "7B": 1.0, "7.8B": 1.0, "8B": 0.9,
                "12B": 0.8, "13B": 0.8, "21.4B": 0.7, "30B": 0.6,
                "32B": 0.6, "70B": 0.5,
            }
            
            for size, bonus in size_map.items():
                if size.upper() in model_size.upper():
                    return bonus
            
            return 1.0  # Default value
    
    def _extract_model_size(self, model_id: str) -> str:
        """Extract model size from model ID"""
        model_id_lower = model_id.lower()
        
        size_patterns = [
            "70b", "32b", "30b", "21.4b", "21-4b",
            "13b", "12b", "8b", "7.8b", "7-8b", "7b",
            "4b", "3b", "2.4b", "2-4b", "2.1b", "2-1b",
            "1.5b", "1-5b", "0.5b", "0-5b", "500m"
        ]
        
        for pattern in size_patterns:
            if pattern in model_id_lower:
                return pattern.upper().replace("-", ".")
        
        return "unknown"
    
    def _estimate_time_by_size(self, model_size: str, task: str) -> float:
        """Estimate execution time by model size and task (seconds)"""
        # Base times (seconds)
        base_times = {
            "0.5B": 300,
            "1.5B": 600,
            "2.1B": 900,
            "2.4B": 1000,
            "3B": 1200,
            "4B": 1800,
            "7B": 3600,
            "7.8B": 3900,
            "8B": 4200,
            "12B": 7200,
            "13B": 7800,
            "21.4B": 14400,
            "30B": 21600,
            "32B": 23400,
            "70B": 43200,
        }
        
        # Task-specific multipliers
        task_multipliers = {
            "mmlu": 2.0,
            "mmlu_pro": 2.5,
            "bbh": 3.0,
            "gsm8k": 1.5,
            "humaneval": 1.2,
            "hellaswag": 1.0,
        }
        
        base_time = 3600  # Default 1 hour
        for size, time in base_times.items():
            if size in model_size.upper():
                base_time = time
                break
        
        # Apply task multiplier
        multiplier = 1.0
        for task_pattern, mult in task_multipliers.items():
            if task_pattern in task.lower():
                multiplier = mult
                break
        
        return base_time * multiplier
    
    def _suggest_batch_size(self, estimated_memory: float) -> int:
        """Suggest batch size based on estimated memory usage"""
        # Check currently available memory
        current_snapshot = self.resource_monitor.get_current_snapshot()
        if current_snapshot:
            available_memory = current_snapshot.gpu_memory_total * (1 - self.memory_safety_margin)
            available_memory -= current_snapshot.gpu_memory_used
            
            if available_memory <= 0:
                return 1
            
            # Estimate memory per batch
            memory_per_batch = estimated_memory
            
            # Calculate safe batch size
            safe_batch_size = int(available_memory / memory_per_batch)
            return max(1, min(safe_batch_size, 8))  # Max 8
        
        # Default values based on adjusted memory
        if estimated_memory < 10:
            return 4
        elif estimated_memory < 20:
            return 2
        else:
            return 1
    
    def _suggest_num_fewshot(self, task: str, estimated_memory: float) -> int:
        """Suggest few-shot number based on task and memory"""
        # Zero-shot tasks
        zero_shot_tasks = ["gpqa", "hrm8k", "haerae", "bbh", "agieval", 
                          "triviaqa", "nq_open", "humaneval", "csatqa"]
        
        if any(zs in task.lower() for zs in zero_shot_tasks):
            return 0
        
        # Adjust few-shot based on memory availability (now uses quantization-adjusted memory)
        if estimated_memory > 45:  # Memory shortage (adjusted threshold)
            return 0
        elif estimated_memory > 25:  # Adjusted threshold
            return 3
        else:
            return 5  # Default value
    
    def _optimize_gpu_assignment(self, schedule: List[TaskPriority]):
        """Optimize GPU assignment"""
        if self.num_gpus == 1:
            # Single GPU case, assign all to GPU 0
            for task in schedule:
                task.suggested_gpu = 0
            return
        
        # Multi-GPU case, distribute load
        gpu_loads = [0.0] * self.num_gpus  # Expected load for each GPU
        
        for task in schedule:
            # Find GPU with minimum load
            min_load_gpu = np.argmin(gpu_loads)
            task.suggested_gpu = min_load_gpu
            
            # Update load for that GPU
            gpu_loads[min_load_gpu] += task.estimated_time
        
        # Log load balance
        logger.info(f"GPU load distribution: {gpu_loads}")
    
    def _generate_rationale(self, model_name: str, task: str, 
                          estimated_time: float, estimated_memory: float,
                          success_probability: float, oom_risk: float,
                          quantization_type: str = None) -> str:
        """Generate scheduling decision rationale"""
        rationale_parts = []
        
        # Time prediction
        hours = estimated_time / 3600
        rationale_parts.append(f"Est. time: {hours:.1f}h")
        
        # Memory prediction with quantization info
        if quantization_type and quantization_type in ['4bit', '8bit']:
            rationale_parts.append(f"Est. memory: {estimated_memory:.1f}GB ({quantization_type})")
        else:
            rationale_parts.append(f"Est. memory: {estimated_memory:.1f}GB")
        
        # Success rate
        rationale_parts.append(f"Success rate: {success_probability*100:.1f}%")
        
        # OOM risk
        if oom_risk > 0.5:
            rationale_parts.append(f"High OOM risk ({oom_risk*100:.0f}%)")
        elif oom_risk > 0.2:
            rationale_parts.append(f"Medium OOM risk ({oom_risk*100:.0f}%)")
        
        return " | ".join(rationale_parts)
    
    def get_next_task(self, schedule: List[TaskPriority], 
                     completed_tasks: List[Tuple[str, str]]) -> Optional[TaskPriority]:
        """Select next task to execute
        
        Args:
            schedule: Complete schedule
            completed_tasks: List of completed (model_id, task_name) tuples
            
        Returns:
            Next task to execute or None
        """
        # Set of completed tasks
        completed_set = set(completed_tasks)
        
        # Check current resource status
        current_resources = self.resource_monitor.get_current_snapshot()
        
        for task in schedule:
            # Skip already completed tasks
            if (task.model_id, task.task_name) in completed_set:
                continue
            
            # Check resource availability
            if self._can_run_task(task, current_resources):
                return task
        
        return None
    
    def _can_run_task(self, task: TaskPriority, 
                     resources: Optional[Any]) -> bool:
        """Check if task can be executed"""
        if not resources:
            return True  # Assume executable if no resource info
        
        # Check GPU memory
        required_memory = task.estimated_memory
        available_memory = resources.gpu_memory_total - resources.gpu_memory_used
        available_memory *= (1 - self.memory_safety_margin)  # Safety margin
        
        if required_memory > available_memory:
            logger.warning(f"Not enough memory for {task.model_id} on {task.task_name}: "
                         f"required={required_memory:.1f}GB, available={available_memory:.1f}GB")
            return False
        
        # Check GPU temperature (optional)
        if resources.gpu_temperature > 85:  # Above 85°C
            logger.warning(f"GPU too hot ({resources.gpu_temperature}°C), waiting...")
            return False
        
        return True
    
    def update_schedule_after_completion(self, schedule: List[TaskPriority],
                                       completed_task: TaskPriority,
                                       actual_time: float,
                                       actual_memory: float,
                                       status: str):
        """Update schedule after task completion
        
        Adjust predictions for remaining tasks based on actual execution results
        """
        if status != "completed":
            # Lower priority for similar tasks if failed
            for task in schedule:
                if (task.model_id == completed_task.model_id and 
                    task.task_name != completed_task.task_name):
                    task.priority_score *= 0.8  # 20% penalty
        
        # Calculate prediction accuracy
        time_error = abs(actual_time - completed_task.estimated_time) / completed_task.estimated_time
        memory_error = abs(actual_memory - completed_task.estimated_memory) / completed_task.estimated_memory
        
        # Re-estimate similar tasks if large error
        if time_error > 0.5 or memory_error > 0.5:
            logger.info(f"Large prediction error for {completed_task.model_id} on {completed_task.task_name}")
            logger.info(f"Time error: {time_error*100:.1f}%, Memory error: {memory_error*100:.1f}%")
            
            # Adjust other tasks from same model
            adjustment_factor = actual_time / completed_task.estimated_time
            for task in schedule:
                if task.model_id == completed_task.model_id:
                    task.estimated_time *= adjustment_factor
                    task.estimated_memory = (task.estimated_memory + actual_memory) / 2
        
        # Re-sort schedule
        schedule.sort(key=lambda x: x.priority_score, reverse=True)
    
    def export_schedule(self, schedule: List[TaskPriority], 
                       filepath: Path):
        """Export schedule to file"""
        # Add timestamp to filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = filepath.stem
        suffix = filepath.suffix
        filepath = filepath.parent / f"{stem}_{timestamp}{suffix}"
        
        data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'num_tasks': len(schedule),
                'num_gpus': self.num_gpus,
                'integrated_with_dynamic_classifier': True,
                'quantization_aware': True,
            },
            'schedule': [
                {
                    'model_id': task.model_id,
                    'task_name': task.task_name,
                    'priority_score': float(task.priority_score),
                    'estimated_time_hours': float(task.estimated_time / 3600),
                    'estimated_memory_gb': float(task.estimated_memory),
                    'success_probability': float(task.success_probability),
                    'suggested_gpu': int(task.suggested_gpu),
                    'suggested_batch_size': int(task.suggested_batch_size),
                    'suggested_num_fewshot': int(task.suggested_num_fewshot),
                    'rationale': task.rationale,
                }
                for task in schedule
            ]
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Schedule exported to {filepath}")