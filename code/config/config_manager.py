"""
Configuration Manager for SpectraBench
Centralized configuration management with environment-specific overrides
"""
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import logging
from datetime import datetime
from typing import Optional, Dict

# Logging setup
logger = logging.getLogger(__name__)


@dataclass
class StageTransitionConfig:
    """Configuration for stage transitions"""
    min_learning_data: int
    stable_learning_data: int
    hybrid_confidence_threshold: float
    dynamic_thresholds: bool = True
    domain_thresholds: Optional[Dict[str, Dict[str, int]]] = None


@dataclass
class MLModelConfig:
    """Configuration for ML models"""
    random_forest: Dict[str, Any]
    training: Dict[str, Any]
    retraining: Dict[str, Any]


@dataclass
class ResourceConfig:
    """Configuration for resource management"""
    memory: Dict[str, Any]
    gpu: Dict[str, Any]
    multi_gpu: Dict[str, Any]


@dataclass
class SchedulingConfig:
    """Configuration for scheduling algorithms"""
    priority_weights: Dict[str, float]
    optimization: Dict[str, Any]
    rollback: Dict[str, Any]


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging"""
    performance: Dict[str, Any]
    resource: Dict[str, Any]
    alerts: Dict[str, Any]


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    threshold_optimization: Dict[str, Any]
    validation: Dict[str, Any]
    ab_testing: Dict[str, Any]


@dataclass
class SystemConfig:
    """Configuration for system settings"""
    database: Dict[str, Any]
    model_persistence: Dict[str, Any]
    logging: Dict[str, Any]


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, 
                 environment: str = "development"):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (development, testing, production)
        """
        self.environment = environment
        
        # Default config path
        if config_path is None:
            # Look for config file in the correct location
            possible_paths = [
                Path(__file__).parent / "scheduler_config.yaml",  # Same directory as config_manager.py
                Path.cwd() / "code" / "config" / "scheduler_config.yaml",  # From project root
                Path(__file__).parent.parent.parent / "code" / "config" / "scheduler_config.yaml"  # Relative path
            ]
            
            config_path = None
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError("scheduler_config.yaml not found. Please ensure it exists in code/config/scheduler_config.yaml")
        
        self.config_path = Path(config_path)
        
        # Load configuration
        self._raw_config = self._load_config()
        self._apply_environment_overrides()
        
        # Parse configuration sections
        self.stage_transitions = self._parse_stage_transitions()
        self.ml_models = self._parse_ml_models()
        self.resource_management = self._parse_resource_management()
        self.scheduling = self._parse_scheduling()
        self.monitoring = self._parse_monitoring()
        self.experiments = self._parse_experiments()
        self.system = self._parse_system()
        
        logger.info(f"Configuration loaded for environment: {environment}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
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
        """Apply environment-specific configuration overrides"""
        if 'environments' in self._raw_config and self.environment in self._raw_config['environments']:
            env_config = self._raw_config['environments'][self.environment]
            self._deep_merge(self._raw_config, env_config)
            logger.info(f"Applied environment overrides for: {self.environment}")
    
    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge two dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _parse_stage_transitions(self) -> StageTransitionConfig:
        """Parse stage transition configuration"""
        config = self._raw_config.get('stage_transitions', {})
        
        return StageTransitionConfig(
            min_learning_data=config.get('min_learning_data', 50),
            stable_learning_data=config.get('stable_learning_data', 200),
            hybrid_confidence_threshold=config.get('hybrid_confidence_threshold', 0.7),
            dynamic_thresholds=config.get('dynamic_thresholds', True),
            domain_thresholds=config.get('domain_thresholds')
        )
    
    def _parse_ml_models(self) -> MLModelConfig:
        """Parse ML model configuration"""
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
        """Parse resource management configuration"""
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
        """Parse scheduling configuration"""
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
        """Parse monitoring configuration"""
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
        """Parse experiment configuration"""
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
        """Parse system configuration"""
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
        """Get domain-specific thresholds for model size"""
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
        
        # Determine domain based on model size
        if model_size in ['small_models', 'medium_models', 'large_models']:
            domain = model_size
        else:
            # Infer domain from model characteristics
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
        """Get model-specific configuration"""
        base_config = self.ml_models.random_forest.copy()
        
        # Adjust parameters based on model size
        if 'large' in model_size.lower():
            base_config['n_estimators'] = min(base_config.get('n_estimators', 100), 50)
            base_config['max_depth'] = min(base_config.get('max_depth', 10), 8)
        elif 'small' in model_size.lower():
            base_config['n_estimators'] = max(base_config.get('n_estimators', 100), 150)
        
        return base_config
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        warnings = []
        
        # Validate stage transitions
        if self.stage_transitions.min_learning_data >= self.stage_transitions.stable_learning_data:
            warnings.append("min_learning_data should be less than stable_learning_data")
        
        # Validate priority weights
        weight_sum = sum(self.scheduling.priority_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            warnings.append(f"Priority weights sum to {weight_sum:.3f}, should be 1.0")
        
        # Validate resource thresholds
        if self.resource_management.memory['safety_margin'] > 0.5:
            warnings.append("Memory safety margin seems too high (>50%)")
        
        # Validate paths
        db_path = Path(self.system.database['path'])
        if not db_path.parent.exists():
            warnings.append(f"Database directory does not exist: {db_path.parent}")
        
        return warnings
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None):
        """Save current configuration to file"""
        if output_path is None:
            output_path = self.config_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self._raw_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def update_thresholds(self, min_learning_data: int, stable_learning_data: int):
        """Update stage transition thresholds"""
        self._raw_config['stage_transitions']['min_learning_data'] = min_learning_data
        self._raw_config['stage_transitions']['stable_learning_data'] = stable_learning_data
        
        # Refresh parsed configuration
        self.stage_transitions = self._parse_stage_transitions()
        
        logger.info(f"Updated thresholds: min_learning_data={min_learning_data}, "
                   f"stable_learning_data={stable_learning_data}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'stage_transitions': self.stage_transitions.__dict__,
            'ml_models': self.ml_models.__dict__,
            'resource_management': self.resource_management.__dict__,
            'scheduling': self.scheduling.__dict__,
            'monitoring': self.monitoring.__dict__,
            'experiments': self.experiments.__dict__,
            'system': self.system.__dict__
        }
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"ConfigManager(environment={self.environment}, " \
               f"min_learning_data={self.stage_transitions.min_learning_data}, " \
               f"stable_learning_data={self.stage_transitions.stable_learning_data})"
               
    def apply_phase_results(self, phase1_result: Optional[Dict] = None, phase2_result: Optional[Dict] = None):
        """Apply Phase 1 and Phase 2 experimental results"""
        if phase1_result:
            # Apply Phase 1 limit convergence results
            recommended_limit = phase1_result.get('recommended_limit')
            if recommended_limit:
                # Update experiment configuration for future use
                if 'experiments' not in self._raw_config:
                    self._raw_config['experiments'] = {}
                if 'validation' not in self._raw_config['experiments']:
                    self._raw_config['experiments']['validation'] = {}
                
                self._raw_config['experiments']['validation']['recommended_limit'] = recommended_limit
                self.experiments = self._parse_experiments()  # Refresh parsed config
                
                logger.info(f"Applied Phase 1 result: recommended_limit={recommended_limit}")
    
        if phase2_result:
            # Apply Phase 2 threshold optimization results
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
                self.experiments = self._parse_experiments()  # Refresh parsed config
            
            logger.info(f"Applied Phase 2 results: "
                    f"min_learning_data={self.stage_transitions.min_learning_data}, "
                    f"stable_learning_data={self.stage_transitions.stable_learning_data}")

def save_config_with_backup(self, output_path: Optional[Union[str, Path]] = None):
    """Save configuration with backup"""
    if output_path is None:
        output_path = self.config_path
    
    output_path = Path(output_path)
    
    # Create backup if original exists
    if output_path.exists():
        backup_path = output_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml')
        import shutil
        shutil.copy2(output_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
    
    # Save current config
    self.save_config(output_path)

def export_config_for_experiments(self, filepath: Path):
    """Export configuration optimized for experiments"""
    # Create experiment-friendly config
    experiment_config = {
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




# Global configuration instance
_config_instance = None


def get_config(environment: str = None) -> ConfigManager:
    """Get global configuration instance"""
    global _config_instance
    
    if _config_instance is None or (environment and _config_instance.environment != environment):
        env = environment or os.getenv('SPECTRABENCH_ENV', 'development')
        _config_instance = ConfigManager(environment=env)
    
    return _config_instance


def reload_config(environment: str = None):
    """Reload configuration"""
    global _config_instance
    env = environment or os.getenv('SPECTRABENCH_ENV', 'development')
    _config_instance = ConfigManager(environment=env)
    return _config_instance


# Convenience functions
def get_stage_thresholds(model_size: str = None) -> Dict[str, int]:
    """Get stage transition thresholds"""
    config = get_config()
    if model_size:
        return config.get_domain_thresholds(model_size)
    return {
        'min_learning_data': config.stage_transitions.min_learning_data,
        'stable_learning_data': config.stage_transitions.stable_learning_data
    }


def get_ml_config(model_size: str = None) -> Dict[str, Any]:
    """Get ML model configuration"""
    config = get_config()
    if model_size:
        return config.get_model_config_for_size(model_size)
    return config.ml_models.random_forest


def get_resource_config() -> ResourceConfig:
    """Get resource management configuration"""
    return get_config().resource_management


def get_scheduling_config() -> SchedulingConfig:
    """Get scheduling configuration"""
    return get_config().scheduling

def update_thresholds_from_experiments(phase1_result: Dict, phase2_result: Dict):
    """Update thresholds based on experimental results"""
    config = get_config()
    config.apply_phase_results(phase1_result, phase2_result)
    config.save_config_with_backup()
    
    logger.info("Thresholds updated from experimental results")

def get_experiment_config() -> Dict[str, Any]:
    """Get experiment-specific configuration"""
    config = get_config()
    return {
        'threshold_ranges': config.experiments.threshold_optimization,
        'validation_settings': config.experiments.validation,
        'ml_settings': config.ml_models.random_forest
    }

if __name__ == "__main__":
    # Test configuration loading
    config = ConfigManager(environment="development")
    
    print("Configuration loaded successfully!")
    print(f"Stage transitions: {config.stage_transitions}")
    print(f"ML model config: {config.ml_models.random_forest}")
    
    # Validate configuration
    warnings = config.validate_config()
    if warnings:
        print("\nConfiguration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nConfiguration validation passed!")
    
    # Test domain-specific thresholds
    print(f"\nSmall model thresholds: {config.get_domain_thresholds('small_models')}")
    print(f"Large model thresholds: {config.get_domain_thresholds('large_models')}")