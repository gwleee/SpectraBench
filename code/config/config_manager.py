"""
Configuration Manager for SpectraBench
Enhanced with complete mode independence and dynamic configuration
"""
import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class StageTransitionConfig:
    min_learning_data: int
    stable_learning_data: int
    hybrid_confidence_threshold: float
    dynamic_thresholds: bool = True
    domain_thresholds: Optional[Dict[str, Dict[str, int]]] = None


@dataclass
class MLModelConfig:
    random_forest: Dict[str, Any]
    training: Dict[str, Any]
    retraining: Dict[str, Any]


@dataclass
class ResourceConfig:
    memory: Dict[str, Any]
    gpu: Dict[str, Any]
    multi_gpu: Dict[str, Any]


@dataclass
class SchedulingConfig:
    priority_weights: Dict[str, float]
    optimization: Dict[str, Any]
    rollback: Dict[str, Any]


@dataclass
class MonitoringConfig:
    performance: Dict[str, Any]
    resource: Dict[str, Any]
    alerts: Dict[str, Any]


@dataclass
class ExperimentConfig:
    threshold_optimization: Dict[str, Any]
    validation: Dict[str, Any]
    ab_testing: Dict[str, Any]


@dataclass
class SystemConfig:
    database: Dict[str, Any]
    model_persistence: Dict[str, Any]
    logging: Dict[str, Any]


class ConfigManager:
    def __init__(self, config_path: Optional[Union[str, Path]] = None, 
                 environment: str = "development", mode: str = None):
        
        if mode is None:
            raise ValueError("Mode must be explicitly specified for proper isolation")
        
        self.environment = environment
        self.mode = mode
        
        process_id = os.getpid()
        thread_id = threading.current_thread().ident
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        
        self.instance_id = f"{environment}_{mode}_{timestamp}_{process_id}_{thread_id}"
        
        if config_path is None:
            possible_paths = [
                Path(__file__).parent / "scheduler_config.yaml",
                Path.cwd() / "code" / "config" / "scheduler_config.yaml",
                Path(__file__).parent.parent.parent / "code" / "config" / "scheduler_config.yaml"
            ]
            
            config_path = None
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError("scheduler_config.yaml not found. Please ensure it exists in code/config/scheduler_config.yaml")
        
        self.config_path = Path(config_path)
        
        self._raw_config = self._load_config()
        self._apply_environment_overrides()
        
        self.stage_transitions = self._parse_stage_transitions()
        self.ml_models = self._parse_ml_models()
        self.resource_management = self._parse_resource_management()
        self.scheduling = self._parse_scheduling()
        self.monitoring = self._parse_monitoring()
        self.experiments = self._parse_experiments()
        self.system = self._parse_system()
        
        logger.info(f"ConfigManager initialized: environment={environment}, mode={mode}, instance={self.instance_id}")
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _apply_environment_overrides(self):
        if 'environments' in self._raw_config and self.environment in self._raw_config['environments']:
            env_config = self._raw_config['environments'][self.environment]
            self._deep_merge(self._raw_config, env_config)
            logger.info(f"Applied environment overrides for: {self.environment}")
    
    def _deep_merge(self, base: Dict, override: Dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _parse_stage_transitions(self) -> StageTransitionConfig:
        config = self._raw_config.get('stage_transitions', {})
        
        return StageTransitionConfig(
            min_learning_data=config.get('min_learning_data', 50),
            stable_learning_data=config.get('stable_learning_data', 200),
            hybrid_confidence_threshold=config.get('hybrid_confidence_threshold', 0.7),
            dynamic_thresholds=config.get('dynamic_thresholds', True),
            domain_thresholds=config.get('domain_thresholds')
        )
    
    def _parse_ml_models(self) -> MLModelConfig:
        config = self._raw_config.get('ml_models', {})
        
        return MLModelConfig(
            random_forest=config.get('random_forest', {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }),
            training=config.get('training', {
                'validation_split': 0.2,
                'cross_validation_folds': 5,
                'early_stopping_patience': 10
            }),
            retraining=config.get('retraining', {
                'interval_hours': 24,
                'min_new_samples': 100,
                'performance_degradation_threshold': 0.05
            })
        )
    
    def _parse_resource_management(self) -> ResourceConfig:
        config = self._raw_config.get('resource_management', {})
        
        return ResourceConfig(
            memory=config.get('memory', {
                'safety_margin': 0.15,
                'oom_prediction_threshold': 0.8,
                'adaptive_batch_sizing': True
            }),
            gpu=config.get('gpu', {
                'utilization_target': 0.8,
                'temperature_limit': 85,
                'memory_fragmentation_threshold': 0.1
            }),
            multi_gpu=config.get('multi_gpu', {
                'load_balancing': True,
                'affinity_optimization': True,
                'communication_overhead_limit': 0.05
            })
        )
    
    def _parse_scheduling(self) -> SchedulingConfig:
        config = self._raw_config.get('scheduling', {})
        
        return SchedulingConfig(
            priority_weights=config.get('priority_weights', {
                'time_efficiency': 0.4,
                'memory_efficiency': 0.3,
                'success_probability': 0.2,
                'resource_utilization': 0.1
            }),
            optimization=config.get('optimization', {
                'algorithm': 'multi_objective',
                'pareto_optimization': True,
                'constraint_satisfaction': True
            }),
            rollback=config.get('rollback', {
                'enabled': True,
                'performance_threshold': 0.9,
                'monitoring_window': 10,
                'cooldown_period': 300
            })
        )
    
    def _parse_monitoring(self) -> MonitoringConfig:
        config = self._raw_config.get('monitoring', {})
        
        return MonitoringConfig(
            performance=config.get('performance', {
                'metrics_collection_interval': 1.0,
                'history_retention_days': 30,
                'detailed_logging': True
            }),
            resource=config.get('resource', {
                'monitoring_interval': 1.0,
                'history_size': 300,
                'alert_thresholds': {
                    'memory_usage': 0.9,
                    'gpu_temperature': 80,
                    'cpu_usage': 0.95
                }
            }),
            alerts=config.get('alerts', {
                'enabled': True,
                'oom_prediction_lead_time': 30,
                'performance_degradation_threshold': 0.1
            })
        )
    
    def _parse_experiments(self) -> ExperimentConfig:
        config = self._raw_config.get('experiments', {})
        
        return ExperimentConfig(
            threshold_optimization=config.get('threshold_optimization', {
                'stage1_range': [25, 50, 75, 100],
                'stage2_range': [100, 150, 200, 250, 300],
                'runs_per_threshold': 3
            }),
            validation=config.get('validation', {
                'statistical_significance': 0.05,
                'confidence_interval': 0.95,
                'minimum_samples': 30
            }),
            ab_testing=config.get('ab_testing', {
                'enabled': False,
                'split_ratio': 0.5,
                'duration_hours': 24
            })
        )
    
    def _parse_system(self) -> SystemConfig:
        config = self._raw_config.get('system', {})
        
        return SystemConfig(
            database=config.get('database', {
                'path': 'data/performanceDB/performance_history.db',
                'backup_interval_hours': 24,
                'cleanup_old_records_days': 90
            }),
            model_persistence=config.get('model_persistence', {
                'save_path': 'data/ml_models',
                'auto_save': True,
                'versioning': True
            }),
            logging=config.get('logging', {
                'level': 'INFO',
                'file': 'logs/scheduler.log',
                'rotation': 'daily',
                'retention_days': 7
            })
        )
    
    def get_domain_thresholds(self, model_size: str) -> Dict[str, int]:
        if not self.stage_transitions.dynamic_thresholds:
            return {
                'min_learning_data': self.stage_transitions.min_learning_data,
                'stable_learning_data': self.stage_transitions.stable_learning_data
            }
        
        if not self.stage_transitions.domain_thresholds:
            return {
                'min_learning_data': self.stage_transitions.min_learning_data,
                'stable_learning_data': self.stage_transitions.stable_learning_data
            }
        
        if model_size in ['small_models', 'medium_models', 'large_models']:
            domain = model_size
        else:
            size_lower = model_size.lower()
            if any(x in size_lower for x in ['0.5b', '1.5b', '2.1b', '3b']):
                domain = 'small_models'
            elif any(x in size_lower for x in ['4b', '7b', '8b']):
                domain = 'medium_models'
            else:
                domain = 'large_models'
        
        domain_config = self.stage_transitions.domain_thresholds.get(domain, {})
        
        return {
            'min_learning_data': domain_config.get('min_learning_data', 
                                                 self.stage_transitions.min_learning_data),
            'stable_learning_data': domain_config.get('stable_learning_data', 
                                                     self.stage_transitions.stable_learning_data)
        }
    
    def get_model_config_for_size(self, model_size: str) -> Dict[str, Any]:
        base_config = self.ml_models.random_forest.copy()
        
        if 'large' in model_size.lower():
            base_config['n_estimators'] = min(base_config.get('n_estimators', 100), 50)
            base_config['max_depth'] = min(base_config.get('max_depth', 10), 8)
        elif 'small' in model_size.lower():
            base_config['n_estimators'] = max(base_config.get('n_estimators', 100), 150)
        
        return base_config
    
    def validate_config(self) -> List[str]:
        warnings = []
        
        if self.stage_transitions.min_learning_data >= self.stage_transitions.stable_learning_data:
            warnings.append("min_learning_data should be less than stable_learning_data")
        
        weight_sum = sum(self.scheduling.priority_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            warnings.append(f"Priority weights sum to {weight_sum:.3f}, should be 1.0")
        
        if self.resource_management.memory['safety_margin'] > 0.5:
            warnings.append("Memory safety margin seems too high (>50%)")
        
        db_path = Path(self.system.database['path'])
        if not db_path.parent.exists():
            warnings.append(f"Database directory does not exist: {db_path.parent}")
        
        return warnings
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None):
        if output_path is None:
            output_path = self.config_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self._raw_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def update_thresholds(self, min_learning_data: int, stable_learning_data: int):
        self._raw_config['stage_transitions']['min_learning_data'] = min_learning_data
        self._raw_config['stage_transitions']['stable_learning_data'] = stable_learning_data
        
        self.stage_transitions = self._parse_stage_transitions()
        
        logger.info(f"Updated thresholds: min_learning_data={min_learning_data}, "
                   f"stable_learning_data={stable_learning_data}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'instance_id': self.instance_id,
            'mode': self.mode,
            'environment': self.environment,
            'stage_transitions': self.stage_transitions.__dict__,
            'ml_models': self.ml_models.__dict__,
            'resource_management': self.resource_management.__dict__,
            'scheduling': self.scheduling.__dict__,
            'monitoring': self.monitoring.__dict__,
            'experiments': self.experiments.__dict__,
            'system': self.system.__dict__
        }
    
    def __str__(self) -> str:
        return f"ConfigManager(environment={self.environment}, mode={self.mode}, " \
               f"instance={self.instance_id}, " \
               f"min_learning_data={self.stage_transitions.min_learning_data}, " \
               f"stable_learning_data={self.stage_transitions.stable_learning_data})"
               
    def apply_phase_results(self, phase1_result: Optional[Dict] = None, phase2_result: Optional[Dict] = None):
        if phase1_result:
            recommended_limit = phase1_result.get('recommended_limit')
            if recommended_limit:
                if 'experiments' not in self._raw_config:
                    self._raw_config['experiments'] = {}
                if 'validation' not in self._raw_config['experiments']:
                    self._raw_config['experiments']['validation'] = {}
                
                self._raw_config['experiments']['validation']['recommended_limit'] = recommended_limit
                self.experiments = self._parse_experiments()
                
                logger.info(f"Applied Phase 1 result: recommended_limit={recommended_limit}")
    
        if phase2_result:
            optimal_config = phase2_result.get('optimal_configuration', {})
            
            if 'stage1_threshold' in optimal_config:
                self.update_thresholds(
                    optimal_config['stage1_threshold'],
                    self.stage_transitions.stable_learning_data
                )
            
            if 'stage2_threshold' in optimal_config:
                self.update_thresholds(
                    self.stage_transitions.min_learning_data,
                    optimal_config['stage2_threshold']
                )
            
            if 'stage1_threshold' in optimal_config and 'stage2_threshold' in optimal_config:
                self.update_thresholds(
                    optimal_config['stage1_threshold'],
                    optimal_config['stage2_threshold']
                )
            
            if 'limit' in optimal_config:
                if 'experiments' not in self._raw_config:
                    self._raw_config['experiments'] = {}
                if 'validation' not in self._raw_config['experiments']:
                    self._raw_config['experiments']['validation'] = {}
                
                self._raw_config['experiments']['validation']['optimal_limit'] = optimal_config['limit']
                self.experiments = self._parse_experiments()
            
            logger.info(f"Applied Phase 2 results: "
                    f"min_learning_data={self.stage_transitions.min_learning_data}, "
                    f"stable_learning_data={self.stage_transitions.stable_learning_data}")

    def save_config_with_backup(self, output_path: Optional[Union[str, Path]] = None):
        if output_path is None:
            output_path = self.config_path
        
        output_path = Path(output_path)
        
        if output_path.exists():
            backup_path = output_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml')
            import shutil
            shutil.copy2(output_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        self.save_config(output_path)

    def export_config_for_experiments(self, filepath: Path):
        experiment_config = {
            'instance_info': {
                'instance_id': self.instance_id,
                'mode': self.mode,
                'environment': self.environment,
                'created_at': datetime.now().isoformat()
            },
            'stage_transitions': {
                'min_learning_data': self.stage_transitions.min_learning_data,
                'stable_learning_data': self.stage_transitions.stable_learning_data,
                'hybrid_confidence_threshold': self.stage_transitions.hybrid_confidence_threshold
            },
            'experiments': {
                'threshold_optimization': self.experiments.threshold_optimization,
                'validation': self.experiments.validation
            },
            'ml_models': {
                'random_forest': self.ml_models.random_forest,
                'retraining': self.ml_models.retraining
            },
            'resource_management': {
                'memory': self.resource_management.memory,
                'gpu': self.resource_management.gpu
            }
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(experiment_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Experiment configuration exported to: {filepath}")


_config_instances = {}
_config_lock = threading.Lock()


def get_config(environment: str = None, mode: str = None) -> ConfigManager:
    if mode is None:
        raise ValueError("Mode must be explicitly specified for proper isolation")
    
    env = environment or "development"
    process_id = os.getpid()
    thread_id = threading.current_thread().ident
    instance_key = f"{env}_{mode}_{process_id}_{thread_id}"
    
    with _config_lock:
        if instance_key not in _config_instances:
            _config_instances[instance_key] = ConfigManager(environment=env, mode=mode)
            logger.info(f"Created new ConfigManager instance for {instance_key}")
    
    return _config_instances[instance_key]


def reload_config(environment: str = None, mode: str = None):
    if mode is None:
        raise ValueError("Mode must be explicitly specified for proper isolation")
    
    env = environment or "development"
    process_id = os.getpid()
    thread_id = threading.current_thread().ident
    instance_key = f"{env}_{mode}_{process_id}_{thread_id}"
    
    with _config_lock:
        _config_instances[instance_key] = ConfigManager(environment=env, mode=mode)
        logger.info(f"Reloaded ConfigManager instance for {instance_key}")
    
    return _config_instances[instance_key]


def cleanup_config_instances(mode: str = None):
    with _config_lock:
        if mode:
            keys_to_remove = [k for k in _config_instances.keys() if f"_{mode}_" in k]
            for key in keys_to_remove:
                del _config_instances[key]
                logger.info(f"Cleaned up config instance: {key}")
        else:
            _config_instances.clear()
            logger.info("Cleared all ConfigManager instances")


def get_stage_thresholds(model_size: str = None, mode: str = None) -> Dict[str, int]:
    if mode is None:
        raise ValueError("Mode must be explicitly specified")
    
    config = get_config(mode=mode)
    if model_size:
        return config.get_domain_thresholds(model_size)
    return {
        'min_learning_data': config.stage_transitions.min_learning_data,
        'stable_learning_data': config.stage_transitions.stable_learning_data
    }


def get_ml_config(model_size: str = None, mode: str = None) -> Dict[str, Any]:
    if mode is None:
        raise ValueError("Mode must be explicitly specified")
    
    config = get_config(mode=mode)
    if model_size:
        return config.get_model_config_for_size(model_size)
    return config.ml_models.random_forest


def get_resource_config(mode: str = None) -> ResourceConfig:
    if mode is None:
        raise ValueError("Mode must be explicitly specified")
    
    return get_config(mode=mode).resource_management


def get_scheduling_config(mode: str = None) -> SchedulingConfig:
    if mode is None:
        raise ValueError("Mode must be explicitly specified")
    
    return get_config(mode=mode).scheduling


def update_thresholds_from_experiments(phase1_result: Dict, phase2_result: Dict, mode: str = None):
    if mode is None:
        raise ValueError("Mode must be explicitly specified")
    
    config = get_config(mode=mode)
    config.apply_phase_results(phase1_result, phase2_result)
    config.save_config_with_backup()
    
    logger.info(f"Thresholds updated from experimental results for mode: {mode}")


def get_experiment_config(mode: str = None) -> Dict[str, Any]:
    if mode is None:
        raise ValueError("Mode must be explicitly specified")
    
    config = get_config(mode=mode)
    return {
        'instance_info': {
            'instance_id': config.instance_id,
            'mode': config.mode,
            'environment': config.environment
        },
        'threshold_ranges': config.experiments.threshold_optimization,
        'validation_settings': config.experiments.validation,
        'ml_settings': config.ml_models.random_forest
    }


if __name__ == "__main__":
    print("Testing mode-specific ConfigManager...")
    
    try:
        baseline_config = ConfigManager(environment="development", mode="baseline")
        print(f"Baseline config: {baseline_config}")
        
        optimized_config = ConfigManager(environment="development", mode="optimized")
        print(f"Optimized config: {optimized_config}")
        
        stage_thresholds_baseline = get_stage_thresholds(mode="baseline")
        stage_thresholds_optimized = get_stage_thresholds(mode="optimized")
        
        print(f"Baseline thresholds: {stage_thresholds_baseline}")
        print(f"Optimized thresholds: {stage_thresholds_optimized}")
        
        baseline_warnings = baseline_config.validate_config()
        optimized_warnings = optimized_config.validate_config()
        
        if baseline_warnings:
            print(f"Baseline warnings: {baseline_warnings}")
        else:
            print("Baseline configuration validation passed!")
            
        if optimized_warnings:
            print(f"Optimized warnings: {optimized_warnings}")
        else:
            print("Optimized configuration validation passed!")
            
        print("Mode-specific configuration testing completed!")
        
    except ValueError as e:
        print(f"Expected error caught: {e}")
        print("This demonstrates proper mode enforcement.")