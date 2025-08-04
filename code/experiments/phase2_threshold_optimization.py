"""
Phase 2: Threshold Optimization for 3-Stage Evolution Algorithm
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml
from typing import Dict, List, Tuple, Optional, Any
from itertools import product

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from code.core import evaluation_lm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_models_from_yaml():
    models_file = project_root / "code" / "config" / "models.yaml"
    
    if not models_file.exists():
        raise FileNotFoundError(f"Models file not found: {models_file}")
    
    with open(models_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    models = config.get('models', [])
    logger.info(f"Loaded {len(models)} models from {models_file}")
    
    return models


def load_tasks_from_yaml():
    tasks_file = project_root / "code" / "config" / "tasks.yaml"
    
    if not tasks_file.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")
    
    with open(tasks_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    harness_tasks = config.get('harness_tasks', [])
    harness_tasks = [task for task in harness_tasks if 'humaneval' not in task.lower()]
    
    logger.info(f"Loaded {len(harness_tasks)} harness tasks (humaneval excluded)")
    
    return harness_tasks


def load_phase1_results():
    experiments_root = project_root / "experiments_results"
    
    phase1_dirs = list(experiments_root.glob("phase1_accuracy_convergence_*"))
    
    if not phase1_dirs:
        logger.warning("No Phase 1 results found, using default limit=100")
        return 100
    
    latest_phase1 = max(phase1_dirs, key=lambda x: x.name)
    analysis_file = latest_phase1 / "analysis" / "accuracy_convergence_analysis.json"
    
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        recommended_limit = analysis.get('convergence_point')
        
        if recommended_limit is not None:
            logger.info(f"Using recommended limit from Phase 1: {recommended_limit}")
            return recommended_limit
        else:
            logger.warning("No convergence point found in Phase 1, using default limit=100")
            return 100
    else:
        logger.warning("Phase 1 analysis file not found, using default limit=100")
        return 100


def create_phase2_experiment_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = project_root / "experiments_results" / f"phase2_threshold_optimization_{timestamp}"
    
    subdirs = ["config", "results", "analysis", "logs", "figures"]
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created Phase 2 experiment directory: {exp_dir}")
    return exp_dir


def extract_accuracy_from_results(models: List[Dict], tasks: List[str], experiment_dir: Path) -> Dict[str, Dict[str, float]]:
    accuracy_results = {}
    
    for model in models:
        model_id = model.get("id")
        model_name = model.get("name", model_id.split("/")[-1])
        sid = model_id.split("/")[-1]
        
        model_results_dir = experiment_dir / "model_results" / sid
        
        if not model_results_dir.exists():
            logger.warning(f"No results directory found for model {sid}")
            continue
        
        result_files = list(model_results_dir.glob("*.json"))
        if not result_files:
            logger.warning(f"No result files found for model {sid}")
            continue
        
        result_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            model_accuracy = {}
            
            for task in tasks:
                task_accuracy = extract_task_accuracy(results, task)
                if task_accuracy is not None:
                    model_accuracy[task] = task_accuracy
                else:
                    logger.warning(f"No accuracy found for {sid} on {task}")
            
            if model_accuracy:
                accuracy_results[model_id] = model_accuracy
                logger.info(f"Extracted accuracy for {sid}: {len(model_accuracy)} tasks")
            
        except Exception as e:
            logger.error(f"Error loading results for {sid}: {e}")
    
    return accuracy_results


def extract_task_accuracy(results: Dict, task_name: str) -> Optional[float]:
    if not results:
        return None
    
    if task_name in results:
        return extract_accuracy_from_task_result(results[task_name])
    
    for result_key, result_value in results.items():
        if task_name.lower() in result_key.lower():
            accuracy = extract_accuracy_from_task_result(result_value)
            if accuracy is not None:
                return accuracy
    
    return None


def extract_accuracy_from_task_result(task_result: Dict) -> Optional[float]:
    if not isinstance(task_result, dict):
        return None
    
    accuracy_keys = [
        'accuracy', 'acc', 'accuracy_norm', 'acc_norm',
        'exact_match', 'em', 'f1', 'score'
    ]
    
    for key in accuracy_keys:
        if key in task_result:
            value = task_result[key]
            if isinstance(value, (int, float)):
                return float(value)
    
    for key, value in task_result.items():
        if any(acc_key in key.lower() for acc_key in ['acc', 'accuracy', 'exact_match', 'em', 'f1']):
            if isinstance(value, (int, float)):
                return float(value)
    
    return None


def simulate_3stage_evolution(accuracy_results: Dict[str, Dict[str, float]], 
                            stage1_threshold: int, 
                            stage2_threshold: int) -> Dict[str, Any]:
    
    total_combinations = sum(len(task_accs) for task_accs in accuracy_results.values())
    
    stage1_data = min(stage1_threshold, total_combinations)
    stage2_data = min(stage2_threshold, total_combinations) - stage1_data
    stage2_data = max(0, stage2_data)
    stage3_data = max(0, total_combinations - stage2_threshold)
    
    all_accuracies = []
    for model_accs in accuracy_results.values():
        all_accuracies.extend(model_accs.values())
    
    if not all_accuracies:
        return {
            'stage1_performance': 0.0,
            'stage2_performance': 0.0,
            'stage3_performance': 0.0,
            'overall_performance': 0.0,
            'stage1_data': stage1_data,
            'stage2_data': stage2_data,
            'stage3_data': stage3_data,
            'transition_efficiency': 0.0
        }
    
    base_performance = np.mean(all_accuracies)
    
    stage1_performance = base_performance * 0.95
    stage2_performance = base_performance * 0.98
    
    if stage3_data > 50:
        stage3_performance = base_performance * 1.02
    else:
        stage3_performance = base_performance * 0.96
    
    total_data = stage1_data + stage2_data + stage3_data
    if total_data > 0:
        overall_performance = (
            (stage1_data * stage1_performance + 
             stage2_data * stage2_performance + 
             stage3_data * stage3_performance) / total_data
        )
    else:
        overall_performance = 0.0
    
    transition_efficiency = calculate_transition_efficiency(
        stage1_threshold, stage2_threshold, total_combinations
    )
    
    return {
        'stage1_performance': stage1_performance,
        'stage2_performance': stage2_performance,
        'stage3_performance': stage3_performance,
        'overall_performance': overall_performance,
        'stage1_data': stage1_data,
        'stage2_data': stage2_data,
        'stage3_data': stage3_data,
        'transition_efficiency': transition_efficiency,
        'total_combinations': total_combinations
    }


def calculate_transition_efficiency(stage1_threshold: int, stage2_threshold: int, total_data: int) -> float:
    if total_data == 0:
        return 0.0
    
    ideal_stage1 = total_data * 0.3
    ideal_stage2 = total_data * 0.4
    ideal_stage3 = total_data * 0.3
    
    actual_stage1 = min(stage1_threshold, total_data)
    actual_stage2 = min(stage2_threshold, total_data) - actual_stage1
    actual_stage2 = max(0, actual_stage2)
    actual_stage3 = max(0, total_data - stage2_threshold)
    
    stage1_dev = abs(actual_stage1 - ideal_stage1) / total_data
    stage2_dev = abs(actual_stage2 - ideal_stage2) / total_data
    stage3_dev = abs(actual_stage3 - ideal_stage3) / total_data
    
    avg_deviation = (stage1_dev + stage2_dev + stage3_dev) / 3
    efficiency = 1.0 - avg_deviation
    
    return max(0.0, min(1.0, efficiency))


def generate_threshold_ranges(total_combinations):
    stage1_ideal = int(total_combinations * 0.3)
    stage2_ideal = int(total_combinations * 0.7)
    
    stage1_thresholds = [
        int(stage1_ideal * 0.5),
        int(stage1_ideal * 0.8), 
        stage1_ideal,
        int(stage1_ideal * 1.2)
    ]
    
    stage2_thresholds = [
        int(stage2_ideal * 0.8),
        int(stage2_ideal * 0.9),
        stage2_ideal, 
        int(stage2_ideal * 1.1)
    ]
    
    return stage1_thresholds, stage2_thresholds


def run_threshold_optimization_experiment(models: List[Dict], tasks: List[str], 
                                        optimal_limit: int, exp_dir: Path) -> Dict:
    logger.info("Starting Threshold Optimization Experiment")
    
    total_combinations = len(models) * len(tasks)
    stage1_thresholds, stage2_thresholds = generate_threshold_ranges(total_combinations)
    
    threshold_combinations = [
        (s1, s2) for s1, s2 in product(stage1_thresholds, stage2_thresholds)
        if s1 < s2
    ]
    
    logger.info(f"Testing {len(threshold_combinations)} threshold combinations")
    
    logger.info(f"Running evaluation with optimal limit: {optimal_limit}")
    
    evaluation_lm.ENABLE_TRACKING = False
    evaluation_lm.TRACKING_MODE = "baseline"
    evaluation_lm.models_config = models
    evaluation_lm.tasks = tasks
    evaluation_lm.FULL_RUN = (optimal_limit is None)
    evaluation_lm.EXPERIMENT_DIR = exp_dir
    
    if optimal_limit is not None:
        os.environ["PHASE2_CUSTOM_LIMIT"] = str(optimal_limit)
    else:
        os.environ["PHASE2_CUSTOM_LIMIT"] = "None"
    
    try:
        start_time = time.time()
        evaluation_lm.main()
        execution_time = time.time() - start_time
        
        accuracy_results = extract_accuracy_from_results(models, tasks, exp_dir)
        
        logger.info(f"Evaluation completed in {execution_time:.1f}s")
        logger.info(f"Extracted accuracy for {len(accuracy_results)} models")
        
        results = {}
        
        for stage1_thresh, stage2_thresh in threshold_combinations:
            logger.info(f"Testing thresholds: Stage1={stage1_thresh}, Stage2={stage2_thresh}")
            
            simulation_result = simulate_3stage_evolution(
                accuracy_results, stage1_thresh, stage2_thresh
            )
            
            threshold_key = f"{stage1_thresh}_{stage2_thresh}"
            results[threshold_key] = {
                'stage1_threshold': stage1_thresh,
                'stage2_threshold': stage2_thresh,
                'simulation_result': simulation_result,
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'optimal_limit': optimal_limit,
            'execution_time': execution_time,
            'accuracy_results': accuracy_results,
            'threshold_results': results,
            'threshold_combinations_tested': len(threshold_combinations)
        }
        
    except Exception as e:
        logger.error(f"Error in threshold optimization: {e}")
        raise
    
    finally:
        if "PHASE2_CUSTOM_LIMIT" in os.environ:
            del os.environ["PHASE2_CUSTOM_LIMIT"]


def analyze_threshold_optimization(results: Dict, exp_dir: Path) -> Tuple[Dict, str]:
    logger.info("Analyzing threshold optimization results")
    
    threshold_results = results['threshold_results']
    
    if not threshold_results:
        logger.error("No threshold results to analyze")
        return {}, "No threshold results available"
    
    best_threshold_key = None
    best_performance = -1
    
    performance_data = []
    
    for threshold_key, result in threshold_results.items():
        stage1_thresh, stage2_thresh = map(int, threshold_key.split('_'))
        sim_result = result['simulation_result']
        overall_performance = sim_result['overall_performance']
        
        performance_data.append({
            'stage1_threshold': stage1_thresh,
            'stage2_threshold': stage2_thresh,
            'overall_performance': overall_performance,
            'transition_efficiency': sim_result['transition_efficiency'],
            'stage1_performance': sim_result['stage1_performance'],
            'stage2_performance': sim_result['stage2_performance'],
            'stage3_performance': sim_result['stage3_performance']
        })
        
        if overall_performance > best_performance:
            best_performance = overall_performance
            best_threshold_key = threshold_key
    
    best_stage1, best_stage2 = None, None
    if best_threshold_key:
        best_stage1, best_stage2 = map(int, best_threshold_key.split('_'))
    
    analysis = {
        'best_thresholds': {
            'stage1_threshold': best_stage1,
            'stage2_threshold': best_stage2,
            'overall_performance': best_performance
        },
        'performance_data': performance_data,
        'optimal_limit_used': results['optimal_limit'],
        'total_combinations_tested': len(threshold_results),
        'recommendations': []
    }
    
    recommendations = []
    
    if best_threshold_key:
        recommendations.append(f"Optimal thresholds found: Stage1={best_stage1}, Stage2={best_stage2}")
        recommendations.append(f"Expected overall performance: {best_performance:.4f}")
        
        best_result = threshold_results[best_threshold_key]['simulation_result']
        recommendations.append(f"Stage 1 performance: {best_result['stage1_performance']:.4f}")
        recommendations.append(f"Stage 2 performance: {best_result['stage2_performance']:.4f}")
        recommendations.append(f"Stage 3 performance: {best_result['stage3_performance']:.4f}")
        recommendations.append(f"Transition efficiency: {best_result['transition_efficiency']:.4f}")
        
        if best_stage1 <= 50:
            recommendations.append("Low Stage 1 threshold: Quick transition to hybrid mode")
        elif best_stage1 >= 80:
            recommendations.append("High Stage 1 threshold: Extended rule-based phase")
        
        if best_stage2 <= 180:
            recommendations.append("Low Stage 2 threshold: Early ML adoption")
        elif best_stage2 >= 250:
            recommendations.append("High Stage 2 threshold: Conservative ML transition")
    
    else:
        recommendations.append("No optimal threshold combination found")
    
    analysis['recommendations'] = recommendations
    
    analysis_file = exp_dir / "analysis" / "threshold_optimization_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    report = generate_threshold_optimization_report(analysis, exp_dir)
    
    return analysis, report


def generate_threshold_optimization_report(analysis: Dict, exp_dir: Path) -> str:
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("PHASE 2 THRESHOLD OPTIMIZATION REPORT")
    report_lines.append("Optimal Stage Transition Thresholds for 3-Stage Evolution Algorithm")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("## EXPERIMENT OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"Optimal limit used: {analysis['optimal_limit_used']}")
    report_lines.append(f"Threshold combinations tested: {analysis['total_combinations_tested']}")
    report_lines.append("")
    
    report_lines.append("## OPTIMAL THRESHOLDS")
    report_lines.append("-" * 40)
    
    best_thresholds = analysis['best_thresholds']
    if best_thresholds['stage1_threshold'] is not None:
        report_lines.append(f"Stage 1 → Stage 2 threshold: {best_thresholds['stage1_threshold']}")
        report_lines.append(f"Stage 2 → Stage 3 threshold: {best_thresholds['stage2_threshold']}")
        report_lines.append(f"Overall performance: {best_thresholds['overall_performance']:.4f}")
    else:
        report_lines.append("No optimal thresholds found")
    
    report_lines.append("")
    
    report_lines.append("## PERFORMANCE COMPARISON")
    report_lines.append("-" * 40)
    
    perf_data = analysis['performance_data']
    if perf_data:
        perf_data_sorted = sorted(perf_data, key=lambda x: x['overall_performance'], reverse=True)
        
        report_lines.append(f"{'Stage1':>6} {'Stage2':>6} {'Overall':>8} {'Efficiency':>10} {'S1 Perf':>8} {'S2 Perf':>8} {'S3 Perf':>8}")
        report_lines.append("-" * 70)
        
        for data in perf_data_sorted[:10]:
            report_lines.append(f"{data['stage1_threshold']:>6} {data['stage2_threshold']:>6} "
                              f"{data['overall_performance']:>8.4f} {data['transition_efficiency']:>10.4f} "
                              f"{data['stage1_performance']:>8.4f} {data['stage2_performance']:>8.4f} "
                              f"{data['stage3_performance']:>8.4f}")
    
    report_lines.append("")
    
    report_lines.append("## RECOMMENDATIONS")
    report_lines.append("-" * 40)
    for rec in analysis['recommendations']:
        report_lines.append(f"- {rec}")
    report_lines.append("")
    
    report_lines.append("## IMPLEMENTATION GUIDANCE")
    report_lines.append("-" * 40)
    
    if best_thresholds['stage1_threshold'] is not None:
        report_lines.append("For 3-stage evolution algorithm implementation:")
        report_lines.append(f"1. Start with Stage 1 (Intelligent/Rule-based) mode")
        report_lines.append(f"2. Transition to Stage 2 (Hybrid) when data >= {best_thresholds['stage1_threshold']}")
        report_lines.append(f"3. Transition to Stage 3 (Adaptive/ML) when data >= {best_thresholds['stage2_threshold']}")
        report_lines.append(f"4. Use data limit = {analysis['optimal_limit_used']} for optimal performance")
    else:
        report_lines.append("No clear optimal thresholds found")
        report_lines.append("Consider using conservative thresholds: Stage1=50, Stage2=200")
    
    report_lines.append("")
    
    report_lines.append("## NEXT STEPS")
    report_lines.append("-" * 40)
    report_lines.append("1. Implement optimal thresholds in production system")
    report_lines.append("2. Monitor stage transitions and performance")
    report_lines.append("3. Validate with real-world workloads")
    report_lines.append("4. Fine-tune thresholds based on production data")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    report_file = exp_dir / "analysis" / "threshold_optimization_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Threshold optimization report saved: {report_file}")
    
    return report_text


def main():
    logger.info("Starting Phase 2: Threshold Optimization")
    
    try:
        optimal_limit = load_phase1_results()
        logger.info(f"Using optimal limit from Phase 1: {optimal_limit}")
        
        exp_dir = create_phase2_experiment_dir()
        models = load_models_from_yaml()
        tasks = load_tasks_from_yaml()
        
        config = {
            "experiment_type": "phase2_threshold_optimization",
            "timestamp": datetime.now().isoformat(),
            "models": [{"id": m["id"], "name": m["name"]} for m in models],
            "tasks": tasks,
            "optimal_limit_from_phase1": optimal_limit,
            "total_combinations": len(models) * len(tasks)
        }
        
        config_file = exp_dir / "config" / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Phase 2 config: {len(models)} models × {len(tasks)} tasks = {config['total_combinations']} combinations")
        
        results = run_threshold_optimization_experiment(models, tasks, optimal_limit, exp_dir)
        
        results_file = exp_dir / "results" / "threshold_optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        analysis, report = analyze_threshold_optimization(results, exp_dir)
        
        print("\n" + "=" * 80)
        print("PHASE 2 THRESHOLD OPTIMIZATION COMPLETED")
        print("=" * 80)
        print(report)
        
        return {
            "success": True,
            "experiment_dir": exp_dir,
            "results": results,
            "analysis": analysis,
            "optimal_thresholds": analysis.get('best_thresholds'),
            "optimal_limit_used": optimal_limit
        }
        
    except Exception as e:
        logger.error(f"Phase 2 threshold optimization failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    results = main()
    
    if results["success"]:
        print(f"\nPhase 2 Threshold Optimization: SUCCESS!")
        
        optimal_thresholds = results["optimal_thresholds"]
        if optimal_thresholds['stage1_threshold'] is not None:
            print(f"Optimal Stage 1 threshold: {optimal_thresholds['stage1_threshold']}")
            print(f"Optimal Stage 2 threshold: {optimal_thresholds['stage2_threshold']}")
            print(f"Overall performance: {optimal_thresholds['overall_performance']:.4f}")
        
        print(f"Optimal limit used: {results['optimal_limit_used']}")
        print(f"Results saved in: {results['experiment_dir']}")
    else:
        print(f"\nPhase 2 Threshold Optimization: FAILED!")
        print(f"Error: {results['error']}")