"""
Scheduler Manager for LLM Evaluation
Manages transition between Intelligent and Adaptive schedulers based on available data
"""
from typing import List, Dict, Optional, Any, Union, Tuple
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Import schedulers
try:
    from .intelligent_scheduler import IntelligentScheduler, TaskPriority
    from .adaptive_scheduler import AdaptiveScheduler, MLPrediction
    from .performance_tracker import PerformanceTracker
    from .resource_monitor import ResourceMonitor
except ImportError:
    from intelligent_scheduler import IntelligentScheduler, TaskPriority
    from adaptive_scheduler import AdaptiveScheduler, MLPrediction
    from performance_tracker import PerformanceTracker
    from resource_monitor import ResourceMonitor

# Import configuration management
try:
    from ..config.config_manager import ConfigManager, get_config, get_stage_thresholds
except ImportError:
    try:
        from code.config.config_manager import ConfigManager, get_config, get_stage_thresholds
    except ImportError:
        # Fallback for when config system is not available
        ConfigManager = None
        get_config = lambda: None
        get_stage_thresholds = lambda: {'min_learning_data': 50, 'stable_learning_data': 200}

# Logging setup
logger = logging.getLogger(__name__)


class SchedulerManager:
    """
    Manages the transition between different scheduling strategies:
    - Cold Start: IntelligentScheduler (rule-based)
    - Warm-up: Hybrid mode (weighted combination)
    - Mature: AdaptiveScheduler (ML-based)
    
    UPDATED: Uses ConfigManager for flexible threshold management
    """
    
    def __init__(self, 
                 performance_tracker: PerformanceTracker,
                 resource_monitor: ResourceMonitor,
                 num_gpus: int = 1,
                 config: Optional[Dict] = None,
                 config_manager: Optional[ConfigManager] = None):
        """
        Args:
            performance_tracker: Performance tracker instance
            resource_monitor: Resource monitor instance
            num_gpus: Number of available GPUs
            config: Legacy configuration parameters (deprecated)
            config_manager: ConfigManager instance (preferred)
        """
        self.performance_tracker = performance_tracker
        self.resource_monitor = resource_monitor
        self.num_gpus = num_gpus
        
        # Configuration management
        self.config_manager = config_manager or self._initialize_config_manager(config)
        self._load_configuration()
        
        # Initialize schedulers
        self.intelligent_scheduler = IntelligentScheduler(
            performance_tracker=performance_tracker,
            resource_monitor=resource_monitor,
            num_gpus=num_gpus
        )
        
        self.adaptive_scheduler = AdaptiveScheduler(
            performance_tracker=performance_tracker,
            resource_monitor=resource_monitor,
            num_gpus=num_gpus
        )
        
        # State tracking
        self.current_mode = None
        self.last_mode_check = None
        self.last_training_time = None
        self.mode_transition_history = []
        
        logger.info(f"SchedulerManager initialized with {num_gpus} GPUs using ConfigManager")
        
        # Initial mode determination
        self._update_current_mode()
    
    def _initialize_config_manager(self, legacy_config: Optional[Dict] = None) -> Optional[ConfigManager]:
        """Initialize ConfigManager with fallback to legacy config"""
        try:
            if ConfigManager is None:
                logger.warning("ConfigManager not available, using legacy configuration")
                return None
            
            config_manager = get_config()
            
            # If legacy config provided, update ConfigManager
            if legacy_config:
                logger.info("Applying legacy configuration overrides")
                if 'min_learning_data' in legacy_config:
                    config_manager.update_thresholds(
                        legacy_config['min_learning_data'],
                        legacy_config.get('stable_learning_data', 
                                        config_manager.stage_transitions.stable_learning_data)
                    )
            
            return config_manager
            
        except Exception as e:
            logger.warning(f"Failed to initialize ConfigManager: {e}, using defaults")
            return None
    
    def _load_configuration(self):
        """Load configuration from ConfigManager or use defaults"""
        if self.config_manager:
            # Use ConfigManager settings
            stage_config = self.config_manager.stage_transitions
            self.min_learning_data = stage_config.min_learning_data
            self.stable_learning_data = stage_config.stable_learning_data
            self.hybrid_confidence_threshold = stage_config.hybrid_confidence_threshold
            self.retrain_interval_hours = self.config_manager.ml_models.retraining['interval_hours']
            
            # Load scheduling configuration
            scheduling_config = self.config_manager.scheduling
            self.priority_weights = scheduling_config.priority_weights
            self.rollback_enabled = scheduling_config.rollback['enabled']
            self.performance_threshold = scheduling_config.rollback['performance_threshold']
            
            logger.info(f"Configuration loaded: min_learning_data={self.min_learning_data}, "
                       f"stable_learning_data={self.stable_learning_data}")
        else:
            # Fallback to hardcoded defaults
            self.min_learning_data = 50
            self.stable_learning_data = 200
            self.hybrid_confidence_threshold = 0.7
            self.retrain_interval_hours = 24
            self.priority_weights = {
                'time_efficiency': 0.4,
                'memory_efficiency': 0.3,
                'success_probability': 0.2,
                'resource_utilization': 0.1
            }
            self.rollback_enabled = True
            self.performance_threshold = 0.9
            
            logger.warning("Using hardcoded default configuration")
    
    def get_domain_specific_thresholds(self, model_id: str) -> Dict[str, int]:
        """Get domain-specific thresholds for a model"""
        if self.config_manager and self.config_manager.stage_transitions.dynamic_thresholds:
            # Extract model size for domain classification
            model_size = self._classify_model_size(model_id)
            return self.config_manager.get_domain_thresholds(model_size)
        else:
            # Return default thresholds
            return {
                'min_learning_data': self.min_learning_data,
                'stable_learning_data': self.stable_learning_data
            }
    
    def _classify_model_size(self, model_id: str) -> str:
        """Classify model size based on model ID"""
        model_id_lower = model_id.lower()
        
        # Small models (≤3B)
        if any(x in model_id_lower for x in ['0.5b', '1.5b', '2.1b', '2.4b', '3b', '500m']):
            return 'small_models'
        # Large models (>8B)
        elif any(x in model_id_lower for x in ['12b', '13b', '21.4b', '30b', '32b', '70b']):
            return 'large_models'
        # Medium models (4-8B)
        else:
            return 'medium_models'
    
    def get_current_mode(self) -> str:
        """Get current scheduling mode"""
        now = datetime.now()
        if (self.last_mode_check is None or 
            (now - self.last_mode_check).seconds > 600):
            self._update_current_mode()
            self.last_mode_check = now
        
        return self.current_mode
    
    def _update_current_mode(self):
        """Update current scheduling mode based on available data"""
        # Get total training records
        total_records = self._get_total_training_records()
        
        thresholds = {
            'min_learning_data': self.min_learning_data,
            'stable_learning_data': self.stable_learning_data
        }
        
        # Determine mode
        if total_records < thresholds['min_learning_data']:
            new_mode = "intelligent"
            reason = f"Insufficient data: {total_records} < {thresholds['min_learning_data']}"
        elif total_records < thresholds['stable_learning_data']:
            new_mode = "hybrid"
            reason = f"Learning phase: {total_records} records"
        else:
            # Check if ML models are trained and confident
            if self.adaptive_scheduler.is_trained():
                confidence = self.adaptive_scheduler.get_model_confidence()
                if confidence >= self.hybrid_confidence_threshold:
                    new_mode = "adaptive"
                    reason = f"ML ready: {total_records} records, confidence: {confidence:.2f}"
                else:
                    new_mode = "hybrid"
                    reason = f"Low confidence: {confidence:.2f} < {self.hybrid_confidence_threshold}"
            else:
                new_mode = "hybrid"
                reason = "ML models not trained"
        
        # Check for rollback conditions
        if self.rollback_enabled and new_mode != "intelligent":
            if self._should_rollback():
                new_mode = self._get_rollback_mode()
                reason += " (rollback triggered)"
        
        # Log mode transition
        if new_mode != self.current_mode:
            if self.current_mode is not None:
                logger.info(f"Scheduler mode transition: {self.current_mode} → {new_mode} ({reason})")
            else:
                logger.info(f"Initial scheduler mode: {new_mode} ({reason})")
            
            # Record transition
            self.mode_transition_history.append({
                'timestamp': datetime.now().isoformat(),
                'from_mode': self.current_mode,
                'to_mode': new_mode,
                'reason': reason,
                'total_records': total_records
            })
            
            self.current_mode = new_mode
        
        # Check if adaptive scheduler needs training/retraining
        self._check_and_train_adaptive()
    
    def _should_rollback(self) -> bool:
        """Check if rollback should be triggered"""
        if not self.rollback_enabled:
            return False
        
        try:
            # Get recent performance metrics
            recent_predictions = self._get_recent_performance_metrics()
            if len(recent_predictions) < 5:  # Need minimum samples
                return False
            
            # Calculate recent success rate
            recent_success_rate = sum(1 for p in recent_predictions if p['success']) / len(recent_predictions)
            
            if recent_success_rate < self.performance_threshold:
                logger.warning(f"Performance below threshold: {recent_success_rate:.2f} < {self.performance_threshold}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rollback conditions: {e}")
            return False
    
    def _get_rollback_mode(self) -> str:
        """Determine which mode to rollback to"""
        # Simple strategy: rollback one level
        if self.current_mode == "adaptive":
            return "hybrid"
        elif self.current_mode == "hybrid":
            return "intelligent"
        else:
            return "intelligent"
    
    def _get_recent_performance_metrics(self) -> List[Dict]:
        """Get recent performance metrics for rollback decisions"""
        try:
            cursor = self.performance_tracker.conn.cursor()
            cursor.execute("""
                SELECT status, timestamp FROM execution_records 
                WHERE timestamp > datetime('now', '-1 hour')
                ORDER BY timestamp DESC LIMIT 20
            """)
            
            records = cursor.fetchall()
            return [{'success': record['status'] == 'completed'} for record in records]
            
        except Exception as e:
            logger.error(f"Error getting recent performance metrics: {e}")
            return []
    
    def _get_total_training_records(self) -> int:
        """Get total number of training records available"""
        try:
            cursor = self.performance_tracker.conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as total_records
                FROM execution_records 
                WHERE status IN ('completed', 'failed', 'oom')
            """)
            
            result = cursor.fetchone()
            return result['total_records'] if result else 0
            
        except Exception as e:
            logger.error(f"Error getting total training records: {e}")
            return 0
    
    def _check_and_train_adaptive(self):
        """Check if adaptive scheduler needs training and train if necessary"""
        try:
            # Check if training is needed
            should_train = False
            
            # Initial training
            if not self.adaptive_scheduler.is_trained():
                total_records = self._get_total_training_records()
                if total_records >= self.min_learning_data:
                    should_train = True
                    reason = f"Initial training with {total_records} records"
            
            # Periodic retraining
            elif self.adaptive_scheduler.should_retrain():
                should_train = True
                reason = "Periodic retraining due to new data"
            
            # Time-based retraining
            elif (self.last_training_time and 
                  (datetime.now() - self.last_training_time).total_seconds() > self.retrain_interval_hours * 3600):
                should_train = True
                reason = f"Time-based retraining after {self.retrain_interval_hours} hours"
            
            # Perform training
            if should_train:
                logger.info(f"Starting adaptive scheduler training: {reason}")
                success = self.adaptive_scheduler.train_models()
                
                if success:
                    self.last_training_time = datetime.now()
                    logger.info("Adaptive scheduler training completed successfully")
                else:
                    logger.warning("Adaptive scheduler training failed")
                    
        except Exception as e:
            logger.error(f"Error in adaptive scheduler training: {e}")
    
    def create_optimal_schedule(self, models: List[Dict], tasks: List[str]) -> List[TaskPriority]:
        """Create optimal schedule using current best scheduler"""
        mode = self.get_current_mode()
        
        logger.info(f"Creating schedule using {mode} mode")
        
        if mode == "intelligent":
            return self._create_intelligent_schedule(models, tasks)
        elif mode == "hybrid":
            return self._create_hybrid_schedule(models, tasks)
        else:  # adaptive
            return self._create_adaptive_schedule(models, tasks)
    
    def _create_intelligent_schedule(self, models: List[Dict], tasks: List[str]) -> List[TaskPriority]:
        """Create schedule using intelligent scheduler only"""
        return self.intelligent_scheduler.create_optimal_schedule(models, tasks)
    
    def _create_adaptive_schedule(self, models: List[Dict], tasks: List[str]) -> List[TaskPriority]:
        """Create schedule using adaptive scheduler only"""
        return self.adaptive_scheduler.create_optimal_schedule(models, tasks)
    
    def _create_hybrid_schedule(self, models: List[Dict], tasks: List[str]) -> List[TaskPriority]:
        """Create schedule using hybrid approach (weighted combination)"""
        # Get schedules from both schedulers
        intelligent_schedule = self.intelligent_scheduler.create_optimal_schedule(models, tasks)
        
        # Only use adaptive if it's trained
        if self.adaptive_scheduler.is_trained():
            adaptive_schedule = self.adaptive_scheduler.create_optimal_schedule(models, tasks)
            confidence = self.adaptive_scheduler.get_model_confidence()
        else:
            adaptive_schedule = []
            confidence = 0.0
        
        # Combine schedules
        if adaptive_schedule and confidence > 0.5:
            combined_schedule = self._combine_schedules(
                intelligent_schedule, adaptive_schedule, confidence
            )
            logger.info(f"Created hybrid schedule with ML confidence: {confidence:.2f}")
        else:
            combined_schedule = intelligent_schedule
            logger.info("Using intelligent schedule in hybrid mode (low ML confidence)")
        
        return combined_schedule
    
    def _combine_schedules(self, 
                          intelligent_schedule: List[TaskPriority], 
                          adaptive_schedule: List[TaskPriority], 
                          ml_confidence: float) -> List[TaskPriority]:
        """Combine schedules from both schedulers using weighted approach"""
        # Create mapping for quick lookup
        adaptive_map = {(tp.model_id, tp.task_name): tp for tp in adaptive_schedule}
        
        combined_schedule = []
        
        for intelligent_task in intelligent_schedule:
            key = (intelligent_task.model_id, intelligent_task.task_name)
            
            if key in adaptive_map:
                adaptive_task = adaptive_map[key]
                
                # Weighted combination of predictions
                ml_weight = ml_confidence
                rule_weight = 1 - ml_confidence
                
                # Combine estimates using configured weights
                combined_time = (rule_weight * intelligent_task.estimated_time + 
                               ml_weight * adaptive_task.estimated_time)
                
                combined_memory = (rule_weight * intelligent_task.estimated_memory + 
                                 ml_weight * adaptive_task.estimated_memory)
                
                combined_success = (rule_weight * intelligent_task.success_probability + 
                                  ml_weight * adaptive_task.success_probability)
                
                # Use ML suggestions for batch size and few-shot if confidence is high
                if ml_confidence > self.hybrid_confidence_threshold:
                    suggested_batch_size = adaptive_task.suggested_batch_size
                    suggested_num_fewshot = adaptive_task.suggested_num_fewshot
                else:
                    suggested_batch_size = intelligent_task.suggested_batch_size
                    suggested_num_fewshot = intelligent_task.suggested_num_fewshot
                
                # Compute combined priority score using configured weights
                combined_priority = self._compute_weighted_priority(
                    combined_time, combined_memory, combined_success, ml_weight
                )
                
                # Create combined rationale
                rationale = f"Hybrid ({ml_weight:.1%} ML): " + \
                           f"Time: {combined_time/3600:.1f}h, " + \
                           f"Memory: {combined_memory:.1f}GB, " + \
                           f"Success: {combined_success*100:.1f}%"
                
                # Create combined task priority
                combined_task = TaskPriority(
                    model_id=intelligent_task.model_id,
                    task_name=intelligent_task.task_name,
                    priority_score=combined_priority,
                    estimated_time=combined_time,
                    estimated_memory=combined_memory,
                    success_probability=combined_success,
                    suggested_gpu=intelligent_task.suggested_gpu,
                    suggested_batch_size=suggested_batch_size,
                    suggested_num_fewshot=suggested_num_fewshot,
                    rationale=rationale
                )
                
                combined_schedule.append(combined_task)
            else:
                # No ML prediction available, use intelligent scheduler result
                intelligent_task.rationale = f"Rule-based: {intelligent_task.rationale}"
                combined_schedule.append(intelligent_task)
        
        # Sort by combined priority
        combined_schedule.sort(key=lambda x: x.priority_score, reverse=True)
        
        return combined_schedule
    
    def _compute_weighted_priority(self, exec_time: float, memory: float, success_prob: float, ml_weight: float) -> float:
        """Compute priority score using configured weights"""
        # Time efficiency component
        time_efficiency = 1 / (1 + exec_time / 3600)  # Normalize to hours
        
        # Memory efficiency component  
        memory_efficiency = 1 / (1 + memory / 40)  # Normalize to 40GB baseline
        
        # Success probability component
        success_component = success_prob
        
        # Resource utilization component (simple approximation)
        resource_utilization = min(time_efficiency * memory_efficiency, 1.0)
        
        # Weighted combination
        priority = (self.priority_weights['time_efficiency'] * time_efficiency +
                   self.priority_weights['memory_efficiency'] * memory_efficiency +
                   self.priority_weights['success_probability'] * success_component +
                   self.priority_weights['resource_utilization'] * resource_utilization)
        
        # Apply ML confidence bonus
        ml_bonus = 1 + (ml_weight * 0.1)  # Up to 10% bonus for high ML confidence
        
        return priority * ml_bonus * 100  # Scale to reasonable range
    
    def get_next_task(self, schedule: List[TaskPriority], 
                     completed_tasks: List[Tuple[str, str]]) -> Optional[TaskPriority]:
        """Get next task to execute with dynamic resource checking"""
        mode = self.get_current_mode()
        
        if mode == "adaptive" and self.adaptive_scheduler.is_trained():
            # Use adaptive scheduler's task selection logic if available
            return self._get_adaptive_next_task(schedule, completed_tasks)
        else:
            # Use intelligent scheduler's task selection
            return self.intelligent_scheduler.get_next_task(schedule, completed_tasks)
    
    def _get_adaptive_next_task(self, schedule: List[TaskPriority], 
                               completed_tasks: List[Tuple[str, str]]) -> Optional[TaskPriority]:
        """Get next task using adaptive scheduler logic"""
        completed_set = set(completed_tasks)
        current_resources = self.resource_monitor.get_current_snapshot()
        
        for task in schedule:
            # Skip completed tasks
            if (task.model_id, task.task_name) in completed_set:
                continue
            
            # Check resource availability with ML prediction refinement
            if self._can_run_task_adaptive(task, current_resources):
                return task
        
        return None
    
    def _can_run_task_adaptive(self, task: TaskPriority, resources: Optional[Any]) -> bool:
        """Check if task can be executed with ML-enhanced resource prediction"""
        if not resources:
            return True
        
        # Base resource check
        if not self.intelligent_scheduler._can_run_task(task, resources):
            return False
        
        # Enhanced check with ML prediction if available
        if self.adaptive_scheduler.is_trained():
            try:
                # Create mock model config for ML prediction
                model_config = {
                    "id": task.model_id,
                    "name": task.model_id.split("/")[-1]
                }
                
                # Get ML prediction
                ml_pred = self.adaptive_scheduler.predict(model_config, task.task_name)
                
                # Use ML prediction for more accurate resource check
                required_memory = ml_pred.memory_usage
                oom_risk = ml_pred.oom_probability
                
                available_memory = resources.gpu_memory_total - resources.gpu_memory_used
                
                # Use configured safety margin
                safety_margin = 0.15
                if self.config_manager:
                    safety_margin = self.config_manager.resource_management.memory['safety_margin']
                
                safety_margin += (oom_risk * 0.1)  # Dynamic safety margin
                available_memory *= (1 - safety_margin)
                
                if required_memory > available_memory:
                    logger.debug(f"ML-enhanced resource check failed for {task.model_id} on {task.task_name}: "
                               f"required={required_memory:.1f}GB, available={available_memory:.1f}GB, "
                               f"OOM risk={oom_risk:.2f}")
                    return False
                
            except Exception as e:
                logger.warning(f"Error in ML-enhanced resource check: {e}")
                # Fall back to basic check
        
        return True
    
    def update_schedule_after_completion(self, schedule: List[TaskPriority],
                                       completed_task: TaskPriority,
                                       actual_time: float,
                                       actual_memory: float,
                                       status: str):
        """Update schedule after task completion with learning"""
        mode = self.get_current_mode()
        
        # Always update intelligent scheduler (for fallback)
        self.intelligent_scheduler.update_schedule_after_completion(
            schedule, completed_task, actual_time, actual_memory, status
        )
        
        # Trigger retraining check if we have new data
        if status in ["completed", "failed", "oom"]:
            self._check_and_train_adaptive()
        
        logger.debug(f"Schedule updated after {status} completion in {mode} mode")
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics"""
        stats = {
            'current_mode': self.current_mode,
            'total_training_records': self._get_total_training_records(),
            'configuration': {
                'min_learning_data': self.min_learning_data,
                'stable_learning_data': self.stable_learning_data,
                'confidence_threshold': self.hybrid_confidence_threshold,
                'rollback_enabled': self.rollback_enabled,
                'priority_weights': self.priority_weights
            },
            'adaptive_status': {
                'is_trained': self.adaptive_scheduler.is_trained(),
                'confidence': self.adaptive_scheduler.get_model_confidence() if self.adaptive_scheduler.is_trained() else 0.0,
                'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
                'should_retrain': self.adaptive_scheduler.should_retrain()
            },
            'mode_transitions': self.mode_transition_history[-10:],  # Last 10 transitions
        }
        
        # Add model performance if available
        if hasattr(self.adaptive_scheduler, 'model_performance'):
            stats['model_performance'] = self.adaptive_scheduler.model_performance
        
        return stats
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update configuration at runtime"""
        if self.config_manager:
            # Update ConfigManager
            if 'min_learning_data' in new_config and 'stable_learning_data' in new_config:
                self.config_manager.update_thresholds(
                    new_config['min_learning_data'],
                    new_config['stable_learning_data']
                )
            
            # Reload configuration
            self._load_configuration()
            
            logger.info("Configuration updated successfully")
        else:
            logger.warning("ConfigManager not available for runtime updates")
    
    def force_mode(self, mode: str, duration_minutes: Optional[int] = None):
        """Force specific scheduling mode for testing/debugging"""
        if mode not in ["intelligent", "hybrid", "adaptive"]:
            raise ValueError(f"Invalid mode: {mode}")
        
        old_mode = self.current_mode
        self.current_mode = mode
        
        logger.warning(f"Forcing scheduler mode: {old_mode} → {mode}")
        
        if duration_minutes:
            # TODO: Implement temporary mode forcing with automatic revert
            logger.info(f"Forced mode will be active for {duration_minutes} minutes")
    
    def export_schedule_with_metadata(self, schedule: List[TaskPriority], filepath: Path):
        """Export schedule with scheduler metadata"""
        # Add timestamp to filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = filepath.stem
        suffix = filepath.suffix
        filepath = filepath.parent / f"{stem}_{timestamp}{suffix}"
        
        # Get comprehensive statistics
        stats = self.get_scheduler_statistics()
        
        data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'scheduler_mode': self.current_mode,
                'num_tasks': len(schedule),
                'num_gpus': self.num_gpus,
                'scheduler_stats': stats,
                'config_manager_available': self.config_manager is not None
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
        
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Schedule with metadata exported to {filepath}")
    
    def get_current_scheduler(self) -> Union[IntelligentScheduler, AdaptiveScheduler]:
        """Get the currently active scheduler instance"""
        mode = self.get_current_mode()
        
        if mode == "adaptive" and self.adaptive_scheduler.is_trained():
            return self.adaptive_scheduler
        else:
            # Return intelligent scheduler for both "intelligent" and "hybrid" modes
            # In hybrid mode, the manager handles the combination logic
            return self.intelligent_scheduler