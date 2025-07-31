"""
Fixed Threshold Optimization Experiment Framework
Added support for limit parameter in ExperimentConfig
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from code.scheduler.scheduler_manager import SchedulerManager
from code.scheduler.performance_tracker import PerformanceTracker
from code.scheduler.resource_monitor import ResourceMonitor
from code.config.config_loader import load_models, load_tasks

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Experiment configuration with limit parameter support"""
    # Threshold ranges to test
    stage1_to_stage2_thresholds: List[int]
    stage2_to_stage3_thresholds: List[int]
    
    # Models and tasks to test
    models: List[Dict]
    tasks: List[str]
    
    # Experiment settings
    num_runs_per_threshold: int = 3
    full_run: bool = False
    limit: Optional[int] = None  # Added limit parameter
    
    # Output settings
    output_dir: Path = Path("experiments_results/threshold_optimization")
    experiment_name: str = "threshold_opt"


@dataclass
class ExperimentResult:
    """Results for a single threshold combination"""
    stage1_threshold: int
    stage2_threshold: int
    run_id: int
    
    # Performance metrics
    total_execution_time: float
    total_success_rate: float
    total_oom_rate: float
    memory_efficiency: float
    
    # Stage transition info
    stage1_tasks: int
    stage2_tasks: int
    stage3_tasks: int
    
    # Detailed metrics
    completed_tasks: int
    failed_tasks: int
    oom_tasks: int
    
    # Timing info
    experiment_start: str
    experiment_end: str
    experiment_duration: float


class ThresholdOptimizer:
    """Main experiment runner for threshold optimization"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results: List[ExperimentResult] = []
        self.experiment_id = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store limit parameter
        self.limit = getattr(config, 'limit', None)
        
        logger.info(f"ThresholdOptimizer initialized: {self.experiment_id}")
        logger.info(f"Using limit: {self.limit}")
    
    def run_full_experiment(self) -> pd.DataFrame:
        """Run complete threshold optimization experiment"""
        logger.info("Starting full threshold optimization experiment")
        
        # Create all threshold combinations
        threshold_combinations = [
            (t1, t2) for t1 in self.config.stage1_to_stage2_thresholds
            for t2 in self.config.stage2_to_stage3_thresholds
            if t1 < t2  # Ensure stage1 < stage2 threshold
        ]
        
        total_experiments = len(threshold_combinations) * self.config.num_runs_per_threshold
        logger.info(f"Total experiments to run: {total_experiments}")
        
        # Run experiments
        experiment_count = 0
        for stage1_thresh, stage2_thresh in threshold_combinations:
            logger.info(f"Testing thresholds: Stage1->2: {stage1_thresh}, Stage2->3: {stage2_thresh}")
            
            for run_id in range(self.config.num_runs_per_threshold):
                experiment_count += 1
                logger.info(f"Running experiment {experiment_count}/{total_experiments}")
                
                result = self._run_single_experiment(stage1_thresh, stage2_thresh, run_id)
                self.results.append(result)
                
                # Save intermediate results
                self._save_intermediate_results()
        
        # Convert to DataFrame and save final results
        results_df = self._convert_results_to_dataframe()
        self._save_final_results(results_df)
        
        return results_df
    
    def _run_single_experiment(self, stage1_thresh: int, stage2_thresh: int, run_id: int) -> ExperimentResult:
        """Run a single experiment with specific thresholds"""
        start_time = datetime.now()
        
        # Create temporary performance tracker for this experiment
        temp_db_path = self.config.output_dir / f"temp_db_{stage1_thresh}_{stage2_thresh}_{run_id}.db"
        performance_tracker = PerformanceTracker(mode="threshold_experiment", db_path=temp_db_path)
        
        # Initialize resource monitor
        resource_monitor = ResourceMonitor(monitoring_interval=1.0)
        resource_monitor.start_monitoring()
        
        try:
            # Create scheduler manager with custom thresholds
            scheduler_manager = SchedulerManager(
                performance_tracker=performance_tracker,
                resource_monitor=resource_monitor,
                num_gpus=1,  # Simplified for experiments
                config={
                    'min_learning_data': stage1_thresh,
                    'stable_learning_data': stage2_thresh,
                    'hybrid_confidence_threshold': 0.7,
                    'retrain_interval_hours': 24
                }
            )
            
            # Track stage transitions
            stage_transitions = {1: 0, 2: 0, 3: 0}
            
            # Simulate execution with the given thresholds
            total_execution_time = 0
            completed_tasks = 0
            failed_tasks = 0
            oom_tasks = 0
            
            # Execute each model-task combination
            for model_idx, model in enumerate(self.config.models):
                for task_idx, task in enumerate(self.config.tasks):
                    
                    # Record task start
                    record_key = performance_tracker.record_start(
                        model_id=model["id"],
                        model_name=model["name"],
                        task_name=task,
                        config={
                            "batch_size": 1,
                            "num_fewshot": 5,
                            "limit": self.limit,  # Use instance limit
                            "threshold_experiment": True
                    }
                    )
                    
                    # Simulate execution (using historical data or estimated values)
                    execution_time, status = self._simulate_task_execution(model, task)
                    total_execution_time += execution_time
                    
                    # Record completion
                    performance_tracker.record_end(
                        record_key=record_key,
                        status=status,
                        results={"simulated": True}
                    )
                    
                    # Update counters
                    if status == "completed":
                        completed_tasks += 1
                    elif status == "oom":
                        oom_tasks += 1
                    else:
                        failed_tasks += 1
                    
                    # Check current scheduler mode
                    current_mode = scheduler_manager.get_current_mode()
                    if current_mode == "intelligent":
                        stage_transitions[1] += 1
                    elif current_mode == "hybrid":
                        stage_transitions[2] += 1
                    else:
                        stage_transitions[3] += 1
            
            # Calculate metrics
            total_tasks = len(self.config.models) * len(self.config.tasks)
            success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            oom_rate = oom_tasks / total_tasks if total_tasks > 0 else 0
            
            # Estimate memory efficiency (simplified)
            memory_efficiency = 0.8 if oom_rate < 0.1 else 0.6
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return ExperimentResult(
                stage1_threshold=stage1_thresh,
                stage2_threshold=stage2_thresh,
                run_id=run_id,
                total_execution_time=total_execution_time,
                total_success_rate=success_rate,
                total_oom_rate=oom_rate,
                memory_efficiency=memory_efficiency,
                stage1_tasks=stage_transitions[1],
                stage2_tasks=stage_transitions[2],
                stage3_tasks=stage_transitions[3],
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                oom_tasks=oom_tasks,
                experiment_start=start_time.isoformat(),
                experiment_end=end_time.isoformat(),
                experiment_duration=duration
            )
            
        finally:
            # Cleanup
            resource_monitor.stop_monitoring()
            performance_tracker.close()
            
            # Remove temporary database
            if temp_db_path.exists():
                temp_db_path.unlink()
    
    def _simulate_task_execution(self, model: Dict, task: str) -> Tuple[float, str]:
        """Simulate task execution based on model and task characteristics with limit consideration"""
        model_id = model["id"]
        
        # Extract model size more accurately
        model_size = self._extract_model_size(model)
        
        # Estimate execution time based on model size and limit
        base_time = self._calculate_base_time(model_size)
        
        # Task-specific multipliers (more realistic)
        task_multipliers = {
            "kmmlu": 1.8,
            "kmmlu_hard": 2.2,
            "haerae": 1.6,
            "kobest": 1.4,
            "csatqa": 1.3,
            "kormedmcqa": 1.9,
            "mmlu": 1.5,
            "arc_challenge": 1.7,
            "arc_easy": 0.9,
            "hellaswag": 0.8,
            "humaneval": 1.1
        }
        
        multiplier = task_multipliers.get(task, 1.0)
        
        # Apply limit factor
        limit_factor = self._calculate_limit_factor(self.config.limit)
        
        execution_time = base_time * multiplier * limit_factor
        
        # Add some randomness
        execution_time *= (0.8 + np.random.random() * 0.4)
        
        # Simulate success/failure with more realistic probabilities
        oom_prob = self._calculate_oom_probability(model_size, self.config.limit)
        
        if np.random.random() < oom_prob:
            return execution_time * 0.3, "oom"
        elif np.random.random() < 0.05:  # 5% general failure rate
            return execution_time * 0.8, "failed"
        
        return execution_time, "completed"
    
    def _extract_model_size(self, model: Dict) -> float:
        """Extract model size from model information"""
        model_name = model.get("name", "")
        model_id = model.get("id", "")
        
        # Size patterns
        size_patterns = {
            "0.5B": 0.5, "1.5B": 1.5, "2.1b": 2.1, "2.4B": 2.4,
            "3b": 3.0, "3B": 3.0, "4b": 4.0, "4B": 4.0,
            "7B": 7.0, "8B": 8.0, "12B": 12.0, "21.4b": 21.4, "32B": 32.0
        }
        
        for pattern, size in size_patterns.items():
            if pattern in model_name or pattern in model_id:
                return size
        
        return 8.0  # Default size
    
    def _calculate_base_time(self, model_size: float) -> float:
        """Calculate base execution time based on model size"""
        if model_size <= 1.0:
            return 150
        elif model_size <= 2.5:
            return 300
        elif model_size <= 4.0:
            return 500
        elif model_size <= 8.0:
            return 900
        elif model_size <= 12.0:
            return 1800
        elif model_size <= 25.0:
            return 3600
        else:
            return 7200
    
    def _calculate_limit_factor(self, limit: Optional[int]) -> float:
        """Calculate limit factor for execution time"""
        if limit is None:
            return 1.0  # Full dataset
        elif limit <= 25:
            return 0.25
        elif limit <= 50:
            return 0.45
        elif limit <= 100:
            return 0.65
        elif limit <= 200:
            return 0.80
        elif limit <= 500:
            return 0.95
        else:
            return 1.0
    
    def _calculate_oom_probability(self, model_size: float, limit: Optional[int]) -> float:
        """Calculate OOM probability based on model size and limit"""
        # Base OOM probability by model size
        if model_size >= 30.0:
            base_prob = 0.25
        elif model_size >= 20.0:
            base_prob = 0.15
        elif model_size >= 10.0:
            base_prob = 0.08
        else:
            base_prob = 0.03
        
        # Adjust by limit
        if limit is None:
            return base_prob * 1.5
        elif limit > 200:
            return base_prob * 1.2
        elif limit > 100:
            return base_prob * 1.0
        else:
            return base_prob * 0.7
    
    def _convert_results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = [asdict(result) for result in self.results]
        return pd.DataFrame(data)
    
    def _save_intermediate_results(self):
        """Save intermediate results during experiment"""
        if not self.results:
            return
        
        results_df = self._convert_results_to_dataframe()
        intermediate_path = self.config.output_dir / f"{self.experiment_id}_intermediate.csv"
        results_df.to_csv(intermediate_path, index=False)
    
    def _save_final_results(self, results_df: pd.DataFrame):
        """Save final experiment results"""
        # Save raw results
        results_path = self.config.output_dir / f"{self.experiment_id}_results.csv"
        results_df.to_csv(results_path, index=False)
        
        # Save aggregated results
        aggregated = self._aggregate_results(results_df)
        aggregated_path = self.config.output_dir / f"{self.experiment_id}_aggregated.csv"
        aggregated.to_csv(aggregated_path, index=False)
        
        # Save best configurations
        best_configs = self._find_best_configurations(aggregated)
        best_path = self.config.output_dir / f"{self.experiment_id}_best_configs.json"
        with open(best_path, 'w') as f:
            json.dump(best_configs, f, indent=2)
        
        logger.info(f"Final results saved:")
        logger.info(f"  Raw results: {results_path}")
        logger.info(f"  Aggregated: {aggregated_path}")
        logger.info(f"  Best configs: {best_path}")
    
    def _aggregate_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate results by threshold combination"""
        aggregated = results_df.groupby(['stage1_threshold', 'stage2_threshold']).agg({
            'total_execution_time': ['mean', 'std'],
            'total_success_rate': ['mean', 'std'],
            'total_oom_rate': ['mean', 'std'],
            'memory_efficiency': ['mean', 'std'],
            'experiment_duration': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() if col[1] else col[0] for col in aggregated.columns]
        
        return aggregated
    
    def _find_best_configurations(self, aggregated_df: pd.DataFrame) -> Dict[str, Any]:
        """Find best threshold configurations"""
        best_configs = {}
        
        # Best for execution time
        best_time_idx = aggregated_df['total_execution_time_mean'].idxmin()
        best_configs['best_execution_time'] = {
            'stage1_threshold': int(aggregated_df.iloc[best_time_idx]['stage1_threshold']),
            'stage2_threshold': int(aggregated_df.iloc[best_time_idx]['stage2_threshold']),
            'execution_time': float(aggregated_df.iloc[best_time_idx]['total_execution_time_mean']),
            'success_rate': float(aggregated_df.iloc[best_time_idx]['total_success_rate_mean']),
            'oom_rate': float(aggregated_df.iloc[best_time_idx]['total_oom_rate_mean'])
        }
        
        # Best for success rate
        best_success_idx = aggregated_df['total_success_rate_mean'].idxmax()
        best_configs['best_success_rate'] = {
            'stage1_threshold': int(aggregated_df.iloc[best_success_idx]['stage1_threshold']),
            'stage2_threshold': int(aggregated_df.iloc[best_success_idx]['stage2_threshold']),
            'execution_time': float(aggregated_df.iloc[best_success_idx]['total_execution_time_mean']),
            'success_rate': float(aggregated_df.iloc[best_success_idx]['total_success_rate_mean']),
            'oom_rate': float(aggregated_df.iloc[best_success_idx]['total_oom_rate_mean'])
        }
        
        # Best for OOM rate (lowest)
        best_oom_idx = aggregated_df['total_oom_rate_mean'].idxmin()
        best_configs['best_oom_rate'] = {
            'stage1_threshold': int(aggregated_df.iloc[best_oom_idx]['stage1_threshold']),
            'stage2_threshold': int(aggregated_df.iloc[best_oom_idx]['stage2_threshold']),
            'execution_time': float(aggregated_df.iloc[best_oom_idx]['total_execution_time_mean']),
            'success_rate': float(aggregated_df.iloc[best_oom_idx]['total_success_rate_mean']),
            'oom_rate': float(aggregated_df.iloc[best_oom_idx]['total_oom_rate_mean'])
        }
        
        # Combined score (weighted)
        aggregated_df['combined_score'] = (
            0.4 * (1 - aggregated_df['total_execution_time_mean'] / aggregated_df['total_execution_time_mean'].max()) +
            0.3 * aggregated_df['total_success_rate_mean'] +
            0.3 * (1 - aggregated_df['total_oom_rate_mean'])
        )
        
        best_combined_idx = aggregated_df['combined_score'].idxmax()
        best_configs['best_combined'] = {
            'stage1_threshold': int(aggregated_df.iloc[best_combined_idx]['stage1_threshold']),
            'stage2_threshold': int(aggregated_df.iloc[best_combined_idx]['stage2_threshold']),
            'execution_time': float(aggregated_df.iloc[best_combined_idx]['total_execution_time_mean']),
            'success_rate': float(aggregated_df.iloc[best_combined_idx]['total_success_rate_mean']),
            'oom_rate': float(aggregated_df.iloc[best_combined_idx]['total_oom_rate_mean']),
            'combined_score': float(aggregated_df.iloc[best_combined_idx]['combined_score'])
        }
        
        return best_configs


def create_default_experiment_config() -> ExperimentConfig:
    """Create default experiment configuration"""
    return ExperimentConfig(
        stage1_to_stage2_thresholds=[25, 50, 75, 100],
        stage2_to_stage3_thresholds=[100, 150, 200, 250, 300],
        models=load_models()[:5],  # Use first 5 models for faster testing
        tasks=load_tasks()["harness"][:5],  # Use first 5 tasks for faster testing
        num_runs_per_threshold=2,  # Reduce for faster testing
        full_run=False,
        limit=None  # Can be set to specific value
    )


def run_threshold_optimization_experiment():
    """Main function to run threshold optimization experiment"""
    logger.info("Starting threshold optimization experiment")
    
    # Create experiment configuration
    config = create_default_experiment_config()
    
    # Create optimizer
    optimizer = ThresholdOptimizer(config)
    
    # Run experiment
    results_df = optimizer.run_full_experiment()
    
    # Display summary
    logger.info("Experiment completed!")
    logger.info(f"Total experiments run: {len(results_df)}")
    logger.info(f"Results saved to: {config.output_dir}")
    
    return results_df


if __name__ == "__main__":
    # Run the experiment
    results = run_threshold_optimization_experiment()
    print("Threshold optimization experiment completed!")
    print(f"Results shape: {results.shape}")
    print("\nBest configurations will be saved in the output directory.")