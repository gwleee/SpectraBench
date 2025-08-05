"""
Fixed Scheduler Manager for LLM Evaluation
Enhanced mode isolation, performance improvements, and stability fixes
FIXED: Deadlock prevention and timeout handling for optimized mode
FIXED: Float to integer conversion errors
"""
from typing import List, Dict, Optional, Any, Union, Tuple
import logging
from pathlib import Path
from datetime import datetime, timedelta
import time
import threading
import sqlite3
import signal

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
        ConfigManager = None
        get_config = lambda mode=None: None
        get_stage_thresholds = lambda: {'min_learning_data': 50, 'stable_learning_data': 200}

logger = logging.getLogger(__name__)


class SafeTimeoutHandler:
    """Safe timeout handler that works across different modes"""
    
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = int(timeout_seconds)
        self.timer = None
        self.timed_out = False
    
    def __enter__(self):
        if hasattr(signal, 'SIGALRM'):
            def timeout_handler(signum, frame):
                self.timed_out = True
                raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")
            
            self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
        else:
            def timeout_callback():
                self.timed_out = True
            
            self.timer = threading.Timer(float(self.timeout_seconds), timeout_callback)
            self.timer.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            if hasattr(self, 'old_handler'):
                signal.signal(signal.SIGALRM, self.old_handler)
        else:
            if self.timer:
                self.timer.cancel()
        
        if self.timed_out and exc_type is None:
            raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")


class SchedulerManager:
    """
    Enhanced Scheduler Manager with complete mode isolation and stability
    """
    
    def __init__(self, 
                 performance_tracker: PerformanceTracker,
                 resource_monitor: ResourceMonitor,
                 num_gpus: int = 1,
                 config: Optional[Dict] = None,
                 config_manager: Optional[ConfigManager] = None):
        """
        Initialize SchedulerManager with mode-specific components
        """
        self.performance_tracker = performance_tracker
        self.resource_monitor = resource_monitor
        self.num_gpus = num_gpus
        
        self.mode = performance_tracker.mode if performance_tracker else "baseline"
        self.run_id_pattern = f"{self.mode}_%"
        
        self.config_manager = config_manager or self._initialize_config_manager(config)
        self._load_configuration()
        
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
        
        self.current_mode = None
        self.last_mode_check = None
        self.last_training_time = None
        self.mode_transition_history = []
        
        self._updating_mode = False
        self._training_in_progress = False
        self._db_operation_lock = threading.Lock()
        self._mode_update_lock = threading.Lock()
        
        if self.mode == "optimized":
            self.db_timeout = 3
            self.mode_check_interval = 180
            self.update_completion_timeout = 10
            self.training_trigger_timeout = 5
        else:
            self.db_timeout = 5
            self.mode_check_interval = 300
            self.update_completion_timeout = 30
            self.training_trigger_timeout = 15
        
        logger.info(f"SchedulerManager initialized: mode={self.mode}, pattern={self.run_id_pattern}, GPUs={num_gpus}")
        
        try:
            self._safe_update_current_mode()
        except Exception as e:
            logger.warning(f"Initial mode update failed: {e}, using intelligent mode")
            self.current_mode = "intelligent"
    
    def _initialize_config_manager(self, legacy_config: Optional[Dict] = None) -> Optional[ConfigManager]:
        """Initialize mode-specific ConfigManager with fallback to legacy config"""
        try:
            if ConfigManager is None:
                logger.warning("ConfigManager not available, using legacy configuration")
                return None
            
            config_manager = get_config(mode=self.mode)
            
            if legacy_config and config_manager:
                logger.info(f"Applying legacy configuration overrides for mode: {self.mode}")
                if 'min_learning_data' in legacy_config:
                    config_manager.update_thresholds(
                        legacy_config['min_learning_data'],
                        legacy_config.get('stable_learning_data', 
                                        config_manager.stage_transitions.stable_learning_data)
                    )
            
            return config_manager
            
        except Exception as e:
            logger.warning(f"Failed to initialize ConfigManager for mode {self.mode}: {e}")
            return None
    
    def _load_configuration(self):
        """Load mode-specific configuration from ConfigManager or use defaults"""
        if self.config_manager:
            stage_config = self.config_manager.stage_transitions
            self.min_learning_data = stage_config.min_learning_data
            self.stable_learning_data = stage_config.stable_learning_data
            self.hybrid_confidence_threshold = stage_config.hybrid_confidence_threshold
            
            ml_config = self.config_manager.ml_models
            self.retrain_interval_hours = ml_config.retraining['interval_hours']
            
            scheduling_config = self.config_manager.scheduling
            self.priority_weights = scheduling_config.priority_weights
            
            rollback_config = scheduling_config.rollback
            if self.mode == "optimized":
                self.rollback_enabled = False
                logger.info("Rollback disabled for optimized mode stability")
            else:
                self.rollback_enabled = rollback_config['enabled']
            
            self.performance_threshold = rollback_config['performance_threshold']
            self.monitoring_window = rollback_config['monitoring_window']
            
            logger.info(f"Configuration loaded for mode {self.mode}: "
                       f"min_learning={self.min_learning_data}, "
                       f"stable_learning={self.stable_learning_data}, "
                       f"rollback_enabled={self.rollback_enabled}")
        else:
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
            self.rollback_enabled = False
            self.performance_threshold = 0.9
            self.monitoring_window = 10
            
            logger.warning(f"Using hardcoded default configuration for mode: {self.mode}")
    
    def get_current_mode(self) -> str:
        """Get current scheduling mode with enhanced thread safety"""
        with self._mode_update_lock:
            if self._updating_mode:
                return self.current_mode or "intelligent"
            
            now = datetime.now()
            if (self.last_mode_check is None or 
                (now - self.last_mode_check).total_seconds() > self.mode_check_interval):
                
                try:
                    if self.mode == "optimized":
                        timeout_val = self.mode_check_interval // 6
                        with SafeTimeoutHandler(timeout_val):
                            self._safe_update_current_mode()
                    else:
                        self._safe_update_current_mode()
                    
                    self.last_mode_check = now
                except (TimeoutError, Exception) as e:
                    logger.error(f"Error updating mode: {e}")
                    if self.current_mode is None:
                        self.current_mode = "intelligent"
            
            return self.current_mode or "intelligent"
    
    def _safe_update_current_mode(self):
        """Safely update current scheduling mode with timeout protection"""
        if self._updating_mode:
            return
        
        try:
            self._updating_mode = True
            
            total_records = self._get_total_training_records_with_strict_isolation()
            
            thresholds = {
                'min_learning_data': self.min_learning_data,
                'stable_learning_data': self.stable_learning_data
            }
            
            if total_records < thresholds['min_learning_data']:
                new_mode = "intelligent"
                reason = f"Insufficient data: {total_records} < {thresholds['min_learning_data']}"
            elif total_records < thresholds['stable_learning_data']:
                new_mode = "hybrid"
                reason = f"Learning phase: {total_records} records"
            else:
                new_mode, reason = self._determine_advanced_mode(total_records)
            
            if self.rollback_enabled and new_mode != "intelligent" and self.mode != "optimized":
                if self._should_rollback_quick_check():
                    new_mode = self._get_rollback_mode()
                    reason += " (rollback triggered)"
            
            if new_mode != self.current_mode:
                if self.current_mode is not None:
                    logger.info(f"Scheduler mode transition: {self.current_mode} -> {new_mode} ({reason})")
                else:
                    logger.info(f"Initial scheduler mode: {new_mode} ({reason})")
                
                self.mode_transition_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'from_mode': self.current_mode,
                    'to_mode': new_mode,
                    'reason': reason,
                    'total_records': total_records
                })
                
                if len(self.mode_transition_history) > 20:
                    self.mode_transition_history = self.mode_transition_history[-20:]
                
                self.current_mode = new_mode
            
            if new_mode in ["hybrid", "adaptive"] and self.mode == "optimized":
                self._maybe_trigger_training_safe()
            elif new_mode in ["hybrid", "adaptive"] and self.mode != "optimized":
                self._maybe_trigger_training()
            
        except Exception as e:
            logger.error(f"Error updating current mode: {e}")
            if self.current_mode is None:
                self.current_mode = "intelligent"
        finally:
            self._updating_mode = False
    
    def _determine_advanced_mode(self, total_records: int) -> Tuple[str, str]:
        """Determine if we should use hybrid or adaptive mode"""
        try:
            is_trained = self._quick_check_if_trained()
            
            if is_trained:
                confidence = self._quick_get_confidence()
                
                if confidence >= self.hybrid_confidence_threshold:
                    return "adaptive", f"ML ready: {total_records} records, confidence: {confidence:.2f}"
                else:
                    return "hybrid", f"Low confidence: {confidence:.2f} < {self.hybrid_confidence_threshold}"
            else:
                return "hybrid", "ML models not trained"
                
        except Exception as e:
            logger.warning(f"Error checking ML status: {e}")
            return "hybrid", "ML status check failed"
    
    def _quick_check_if_trained(self) -> bool:
        """Quick non-blocking check if adaptive scheduler is trained"""
        try:
            return self.adaptive_scheduler.is_trained()
        except Exception:
            return False
    
    def _quick_get_confidence(self) -> float:
        """Quick non-blocking confidence check"""
        try:
            return self.adaptive_scheduler.get_model_confidence()
        except Exception:
            return 0.0
    
    def _should_rollback_quick_check(self) -> bool:
        """Quick rollback check without blocking operations"""
        if not self.rollback_enabled:
            return False
        
        try:
            with self._db_operation_lock:
                cursor = self.performance_tracker.conn.cursor()
                cursor.execute("PRAGMA busy_timeout = 1000")
                
                cursor.execute("""
                    SELECT COUNT(*) as failures 
                    FROM execution_records 
                    WHERE mode = ? AND run_id LIKE ? 
                    AND timestamp > datetime('now', '-1 hour')
                    AND status IN ('failed', 'oom')
                    LIMIT 10
                """, (self.mode, self.run_id_pattern))
                
                result = cursor.fetchone()
                failure_count = result['failures'] if result else 0
                
                return failure_count > 3
                
        except (sqlite3.OperationalError, Exception) as e:
            logger.debug(f"Quick rollback check failed: {e}")
            return False
    
    def _get_rollback_mode(self) -> str:
        """Determine which mode to rollback to"""
        if self.current_mode == "adaptive":
            return "hybrid"
        elif self.current_mode == "hybrid":
            return "intelligent"
        else:
            return "intelligent"
    
    def _get_total_training_records_with_strict_isolation(self) -> int:
        """Get total training records with strict mode isolation using run_id pattern"""
        try:
            with self._db_operation_lock:
                cursor = self.performance_tracker.conn.cursor()
                timeout_ms = self.db_timeout * 1000
                cursor.execute(f"PRAGMA busy_timeout = {timeout_ms}")
                
                cursor.execute("""
                    SELECT COUNT(*) as total_records
                    FROM execution_records 
                    WHERE status IN ('completed', 'failed', 'oom')
                    AND mode = ? AND run_id LIKE ?
                """, (self.mode, self.run_id_pattern))
                
                result = cursor.fetchone()
                count = result['total_records'] if result else 0
                
                logger.debug(f"Training records for mode {self.mode} with pattern {self.run_id_pattern}: {count}")
                return count
                
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                logger.warning(f"Database locked during record count for mode {self.mode}")
                return 0
            else:
                logger.error(f"Database error getting training records: {e}")
                return 0
        except Exception as e:
            logger.warning(f"Error getting total training records: {e}")
            return 0
    
    def _maybe_trigger_training(self):
        """Maybe trigger adaptive scheduler training (non-blocking) - ORIGINAL for baseline"""
        if self._training_in_progress:
            return
        
        try:
            if not self._quick_check_if_trained():
                total_records = self._get_total_training_records_with_strict_isolation()
                
                if total_records >= self.min_learning_data:
                    training_thread = threading.Thread(
                        target=self._background_training,
                        args=(total_records,),
                        daemon=True
                    )
                    training_thread.start()
                    logger.info(f"Background training started with {total_records} records")
                    
        except Exception as e:
            logger.warning(f"Error checking training needs: {e}")
    
    def _maybe_trigger_training_safe(self):
        """Safe training trigger for optimized mode with timeout"""
        if self._training_in_progress:
            return
        
        try:
            with SafeTimeoutHandler(self.training_trigger_timeout):
                if not self._quick_check_if_trained():
                    total_records = self._get_total_training_records_with_strict_isolation()
                    
                    if total_records >= self.min_learning_data:
                        training_thread = threading.Thread(
                            target=self._background_training_safe,
                            args=(total_records,),
                            daemon=True
                        )
                        training_thread.start()
                        logger.info(f"Safe background training started with {total_records} records")
                        
        except TimeoutError:
            logger.warning(f"Training trigger timed out after {self.training_trigger_timeout} seconds")
        except Exception as e:
            logger.warning(f"Error in safe training trigger: {e}")
    
    def _background_training(self, total_records: int):
        """Background training to avoid blocking main thread - ORIGINAL for baseline"""
        if self._training_in_progress:
            return
        
        try:
            self._training_in_progress = True
            logger.info(f"Starting background adaptive scheduler training with {total_records} records")
            
            training_success = False
            start_time = time.time()
            
            try:
                training_success = self.adaptive_scheduler.train_models()
                training_time = time.time() - start_time
                
                if training_success:
                    self.last_training_time = datetime.now()
                    logger.info(f"Adaptive scheduler training completed successfully in {training_time:.1f}s")
                else:
                    logger.warning("Adaptive scheduler training failed")
                    
            except Exception as train_error:
                logger.error(f"Training failed: {train_error}")
                
        except Exception as e:
            logger.error(f"Error in background training: {e}")
        finally:
            self._training_in_progress = False
    
    def _background_training_safe(self, total_records: int):
        """Safe background training for optimized mode with enhanced error handling"""
        if self._training_in_progress:
            return
        
        try:
            self._training_in_progress = True
            logger.info(f"Starting SAFE background adaptive scheduler training with {total_records} records")
            
            training_success = False
            start_time = time.time()
            max_training_time = 300
            
            try:
                with SafeTimeoutHandler(max_training_time):
                    training_success = self.adaptive_scheduler.train_models()
                    training_time = time.time() - start_time
                    
                    if training_success:
                        self.last_training_time = datetime.now()
                        logger.info(f"SAFE adaptive scheduler training completed successfully in {training_time:.1f}s")
                    else:
                        logger.warning("SAFE adaptive scheduler training failed")
                        
            except TimeoutError:
                logger.error(f"SAFE training timed out after {max_training_time} seconds")
            except Exception as train_error:
                logger.error(f"SAFE training failed: {train_error}")
                
        except Exception as e:
            logger.error(f"Error in SAFE background training: {e}")
        finally:
            self._training_in_progress = False
    
    def create_optimal_schedule(self, models: List[Dict], tasks: List[str]) -> List[TaskPriority]:
        """Create optimal schedule using current best scheduler"""
        mode = self.get_current_mode()
        
        logger.info(f"Creating schedule using {mode} mode for {len(models)} models and {len(tasks)} tasks")
        
        try:
            if mode == "intelligent":
                return self._create_intelligent_schedule(models, tasks)
            elif mode == "hybrid":
                return self._create_hybrid_schedule(models, tasks)
            else:
                return self._create_adaptive_schedule(models, tasks)
        except Exception as e:
            logger.error(f"Error creating schedule in {mode} mode: {e}")
            logger.info("Falling back to intelligent scheduler")
            return self._create_intelligent_schedule(models, tasks)
    
    def _create_intelligent_schedule(self, models: List[Dict], tasks: List[str]) -> List[TaskPriority]:
        """Create schedule using intelligent scheduler only"""
        return self.intelligent_scheduler.create_optimal_schedule(models, tasks)
    
    def _create_adaptive_schedule(self, models: List[Dict], tasks: List[str]) -> List[TaskPriority]:
        """Create schedule using adaptive scheduler only"""
        try:
            return self.adaptive_scheduler.create_optimal_schedule(models, tasks)
        except Exception as e:
            logger.error(f"Adaptive scheduler failed: {e}, falling back to intelligent")
            return self._create_intelligent_schedule(models, tasks)
    
    def _create_hybrid_schedule(self, models: List[Dict], tasks: List[str]) -> List[TaskPriority]:
        """Create schedule using hybrid approach (weighted combination)"""
        intelligent_schedule = self.intelligent_scheduler.create_optimal_schedule(models, tasks)
        
        adaptive_schedule = []
        confidence = 0.0
        
        try:
            if self._quick_check_if_trained():
                adaptive_schedule = self.adaptive_scheduler.create_optimal_schedule(models, tasks)
                confidence = self._quick_get_confidence()
                logger.info(f"Using adaptive predictions with confidence: {confidence:.2f}")
            else:
                logger.info("ML models not trained, using intelligent schedule only")
        except Exception as e:
            logger.warning(f"Error getting adaptive schedule: {e}")
        
        if adaptive_schedule and confidence > 0.5:
            try:
                combined_schedule = self._combine_schedules(
                    intelligent_schedule, adaptive_schedule, confidence
                )
                logger.info(f"Created hybrid schedule with ML confidence: {confidence:.2f}")
                return combined_schedule
            except Exception as e:
                logger.error(f"Error combining schedules: {e}")
        
        logger.info("Using intelligent schedule in hybrid mode")
        return intelligent_schedule
    
    def _combine_schedules(self, 
                          intelligent_schedule: List[TaskPriority], 
                          adaptive_schedule: List[TaskPriority], 
                          ml_confidence: float) -> List[TaskPriority]:
        """Combine schedules from both schedulers using weighted approach"""
        adaptive_map = {(tp.model_id, tp.task_name): tp for tp in adaptive_schedule}
        
        combined_schedule = []
        
        for intelligent_task in intelligent_schedule:
            key = (intelligent_task.model_id, intelligent_task.task_name)
            
            if key in adaptive_map:
                adaptive_task = adaptive_map[key]
                
                ml_weight = ml_confidence
                rule_weight = 1 - ml_confidence
                
                combined_time = (rule_weight * intelligent_task.estimated_time + 
                               ml_weight * adaptive_task.estimated_time)
                
                combined_memory = (rule_weight * intelligent_task.estimated_memory + 
                                 ml_weight * adaptive_task.estimated_memory)
                
                combined_success = (rule_weight * intelligent_task.success_probability + 
                                  ml_weight * adaptive_task.success_probability)
                
                if ml_confidence > self.hybrid_confidence_threshold:
                    suggested_batch_size = adaptive_task.suggested_batch_size
                    suggested_num_fewshot = adaptive_task.suggested_num_fewshot
                else:
                    suggested_batch_size = intelligent_task.suggested_batch_size
                    suggested_num_fewshot = intelligent_task.suggested_num_fewshot
                
                combined_priority = self._compute_weighted_priority(
                    combined_time, combined_memory, combined_success, ml_weight
                )
                
                rationale = f"Hybrid ({ml_weight:.1%} ML): " + \
                           f"Time: {combined_time/3600:.1f}h, " + \
                           f"Memory: {combined_memory:.1f}GB, " + \
                           f"Success: {combined_success*100:.1f}%"
                
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
                intelligent_task.rationale = f"Rule-based: {intelligent_task.rationale}"
                combined_schedule.append(intelligent_task)
        
        combined_schedule.sort(key=lambda x: x.priority_score, reverse=True)
        
        return combined_schedule
    
    def _compute_weighted_priority(self, exec_time: float, memory: float, success_prob: float, ml_weight: float) -> float:
        """Compute priority score using configured weights"""
        time_efficiency = 1 / (1 + exec_time / 3600)
        memory_efficiency = 1 / (1 + memory / 40)
        success_component = success_prob
        resource_utilization = min(time_efficiency * memory_efficiency, 1.0)
        
        priority = (self.priority_weights['time_efficiency'] * time_efficiency +
                   self.priority_weights['memory_efficiency'] * memory_efficiency +
                   self.priority_weights['success_probability'] * success_component +
                   self.priority_weights['resource_utilization'] * resource_utilization)
        
        ml_bonus = 1 + (ml_weight * 0.1)
        
        return priority * ml_bonus * 100
    
    def get_next_task(self, schedule: List[TaskPriority], 
                     completed_tasks: List[Tuple[str, str]]) -> Optional[TaskPriority]:
        """Get next task to execute with resource checking"""
        mode = self.get_current_mode()
        
        try:
            if mode == "adaptive" and self._quick_check_if_trained():
                return self._get_adaptive_next_task(schedule, completed_tasks)
            else:
                return self.intelligent_scheduler.get_next_task(schedule, completed_tasks)
        except Exception as e:
            logger.error(f"Error getting next task: {e}")
            return self.intelligent_scheduler.get_next_task(schedule, completed_tasks)
    
    def _get_adaptive_next_task(self, schedule: List[TaskPriority], 
                               completed_tasks: List[Tuple[str, str]]) -> Optional[TaskPriority]:
        """Get next task using adaptive scheduler logic"""
        completed_set = set(completed_tasks)
        
        try:
            current_resources = self.resource_monitor.get_current_snapshot()
        except Exception as e:
            logger.warning(f"Error getting resource snapshot: {e}")
            current_resources = None
        
        for task in schedule:
            if (task.model_id, task.task_name) in completed_set:
                continue
            
            if self._can_run_task_enhanced(task, current_resources):
                return task
        
        return None
    
    def _can_run_task_enhanced(self, task: TaskPriority, resources: Optional[Any]) -> bool:
        """Enhanced resource availability check with ML predictions"""
        if not resources:
            return True
        
        try:
            if not self.intelligent_scheduler._can_run_task(task, resources):
                return False
        except Exception as e:
            logger.warning(f"Error in base resource check: {e}")
            return True
        
        try:
            if self._quick_check_if_trained():
                model_config = {
                    "id": task.model_id,
                    "name": task.model_id.split("/")[-1]
                }
                
                ml_pred = self.adaptive_scheduler.predict(model_config, task.task_name)
                
                required_memory = ml_pred.memory_usage
                oom_risk = ml_pred.oom_probability
                
                available_memory = resources.gpu_memory_total - resources.gpu_memory_used
                
                base_safety_margin = 0.15
                if self.config_manager:
                    base_safety_margin = self.config_manager.resource_management.memory['safety_margin']
                
                dynamic_safety_margin = base_safety_margin + (oom_risk * 0.1)
                available_memory *= (1 - dynamic_safety_margin)
                
                if required_memory > available_memory:
                    logger.debug(f"ML-enhanced resource check failed for {task.model_id} on {task.task_name}: "
                               f"required={required_memory:.1f}GB, available={available_memory:.1f}GB, "
                               f"OOM risk={oom_risk:.2f}")
                    return False
                
        except Exception as e:
            logger.warning(f"Error in ML-enhanced resource check: {e}")
        
        return True
    
    def update_schedule_after_completion(self, schedule: List[TaskPriority],
                                       completed_task: TaskPriority,
                                       actual_time: float,
                                       actual_memory: float,
                                       status: str):
        """Update schedule after task completion with enhanced safety for optimized mode"""
        mode = self.get_current_mode()
        
        try:
            if self.mode == "optimized":
                with SafeTimeoutHandler(self.update_completion_timeout):
                    self._update_schedule_completion_safe(schedule, completed_task, actual_time, actual_memory, status)
            else:
                self._update_schedule_completion_original(schedule, completed_task, actual_time, actual_memory, status)
            
            logger.debug(f"Schedule updated after {status} completion in {mode} mode")
            
        except TimeoutError:
            logger.error(f"Schedule update timed out after {self.update_completion_timeout} seconds in {self.mode} mode")
        except Exception as e:
            logger.error(f"Error updating schedule after completion: {e}")
    
    def _update_schedule_completion_original(self, schedule: List[TaskPriority],
                                           completed_task: TaskPriority,
                                           actual_time: float,
                                           actual_memory: float,
                                           status: str):
        """Original update logic for baseline mode"""
        self.intelligent_scheduler.update_schedule_after_completion(
            schedule, completed_task, actual_time, actual_memory, status
        )
        
        if status in ["completed", "failed", "oom"]:
            self._maybe_trigger_training()
    
    def _update_schedule_completion_safe(self, schedule: List[TaskPriority],
                                       completed_task: TaskPriority,
                                       actual_time: float,
                                       actual_memory: float,
                                       status: str):
        """Safe update logic for optimized mode with timeout protection"""
        self.intelligent_scheduler.update_schedule_after_completion(
            schedule, completed_task, actual_time, actual_memory, status
        )
        
        if status in ["completed", "failed", "oom"]:
            try:
                self._maybe_trigger_training_safe()
            except Exception as e:
                logger.warning(f"Safe training trigger failed: {e}")
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics with error handling"""
        try:
            stats = {
                'current_mode': self.current_mode,
                'performance_tracker_mode': self.mode,
                'run_id_pattern': self.run_id_pattern,
                'total_training_records': self._get_total_training_records_with_strict_isolation(),
                'configuration': {
                    'min_learning_data': self.min_learning_data,
                    'stable_learning_data': self.stable_learning_data,
                    'confidence_threshold': self.hybrid_confidence_threshold,
                    'rollback_enabled': self.rollback_enabled,
                    'priority_weights': self.priority_weights
                },
                'adaptive_status': {
                    'is_trained': False,
                    'confidence': 0.0,
                    'last_training': None,
                    'should_retrain': False
                },
                'mode_transitions': self.mode_transition_history[-10:],
            }
            
            try:
                stats['adaptive_status'] = {
                    'is_trained': self._quick_check_if_trained(),
                    'confidence': self._quick_get_confidence(),
                    'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
                    'should_retrain': False
                }
            except Exception as e:
                logger.warning(f"Error getting adaptive status: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting scheduler statistics: {e}")
            return {
                'current_mode': self.current_mode or 'intelligent',
                'performance_tracker_mode': self.mode,
                'run_id_pattern': self.run_id_pattern,
                'error': str(e),
                'total_training_records': 0,
                'configuration': {},
                'adaptive_status': {'is_trained': False, 'confidence': 0.0},
                'mode_transitions': []
            }
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update configuration at runtime"""
        if self.config_manager:
            try:
                if 'min_learning_data' in new_config and 'stable_learning_data' in new_config:
                    self.config_manager.update_thresholds(
                        new_config['min_learning_data'],
                        new_config['stable_learning_data']
                    )
                
                self._load_configuration()
                
                logger.info(f"Configuration updated successfully for mode: {self.mode}")
            except Exception as e:
                logger.error(f"Error updating configuration: {e}")
        else:
            logger.warning("ConfigManager not available for runtime updates")
    
    def force_mode(self, mode: str, duration_minutes: Optional[int] = None):
        """Force specific scheduling mode for testing or debugging"""
        if mode not in ["intelligent", "hybrid", "adaptive"]:
            raise ValueError(f"Invalid mode: {mode}")
        
        old_mode = self.current_mode
        self.current_mode = mode
        
        logger.warning(f"Forcing scheduler mode: {old_mode} -> {mode}")
        
        if duration_minutes:
            logger.info(f"Forced mode will be active for {duration_minutes} minutes")
    
    def export_schedule_with_metadata(self, schedule: List[TaskPriority], filepath: Path):
        """Export schedule with comprehensive scheduler metadata"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stem = filepath.stem
            suffix = filepath.suffix
            filepath = filepath.parent / f"{stem}_{timestamp}{suffix}"
            
            stats = self.get_scheduler_statistics()
            
            data = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'scheduler_mode': self.current_mode,
                    'performance_tracker_mode': self.mode,
                    'run_id_pattern': self.run_id_pattern,
                    'num_tasks': len(schedule),
                    'num_gpus': self.num_gpus,
                    'scheduler_stats': stats,
                    'config_manager_available': self.config_manager is not None,
                    'deadlock_fixes_applied': self.mode == "optimized"
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
            
        except Exception as e:
            logger.error(f"Error exporting schedule: {e}")
    
    def get_current_scheduler(self) -> Union[IntelligentScheduler, AdaptiveScheduler]:
        """Get the currently active scheduler instance"""
        mode = self.get_current_mode()
        
        try:
            if mode == "adaptive" and self._quick_check_if_trained():
                return self.adaptive_scheduler
            else:
                return self.intelligent_scheduler
        except Exception as e:
            logger.error(f"Error getting current scheduler: {e}")
            return self.intelligent_scheduler
    
    def cleanup(self):
        """Cleanup resources and background threads"""
        try:
            self._training_in_progress = False
            self.mode_transition_history.clear()
            
            logger.info(f"SchedulerManager cleanup completed for mode: {self.mode}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass