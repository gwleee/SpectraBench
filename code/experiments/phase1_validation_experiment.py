"""
Phase 1: Accuracy Convergence Analysis for Limit Setting 
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


def create_accuracy_experiment_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = project_root / "experiments_results" / f"phase1_accuracy_convergence_{timestamp}"
    
    subdirs = ["config", "results", "logs", "analysis", "figures"]
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created accuracy convergence experiment directory: {exp_dir}")
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


def run_accuracy_convergence_experiment(models: List[Dict], tasks: List[str], test_limits: List[Optional[int]], exp_dir: Path) -> Dict:
    logger.info("Starting Accuracy Convergence Experiment")
    
    results = {}
    
    for limit_idx, limit in enumerate(test_limits):
        logger.info(f"Testing limit {limit_idx + 1}/{len(test_limits)}: {limit}")
        
        evaluation_lm.ENABLE_TRACKING = False
        evaluation_lm.TRACKING_MODE = "baseline"
        evaluation_lm.models_config = models
        evaluation_lm.tasks = tasks
        evaluation_lm.FULL_RUN = (limit is None)
        evaluation_lm.EXPERIMENT_DIR = exp_dir
        
        if limit is not None:
            os.environ["PHASE1_CUSTOM_LIMIT"] = str(limit)
            logger.info(f"Set PHASE1_CUSTOM_LIMIT = {limit}")
        else:
            os.environ["PHASE1_CUSTOM_LIMIT"] = "None"
            logger.info("Set PHASE1_CUSTOM_LIMIT = None (full dataset)")
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting evaluation with limit = {limit}")
            evaluation_lm.main()
            
            logger.info("Extracting accuracy results...")
            accuracy_results = extract_accuracy_from_results(models, tasks, exp_dir)
            
            limit_stats = calculate_limit_statistics(accuracy_results, limit)
            
            execution_time = time.time() - start_time
            
            results[limit] = {
                'limit': limit,
                'execution_time': execution_time,
                'accuracy_results': accuracy_results,
                'statistics': limit_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Limit {limit} completed in {execution_time:.1f}s")
            logger.info(f"Average accuracy: {limit_stats['avg_accuracy']:.4f}")
            logger.info(f"Successful model-task pairs: {limit_stats['num_successful_pairs']}")
            
        except Exception as e:
            logger.error(f"Error testing limit {limit}: {e}")
            results[limit] = {
                'limit': limit,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            if "PHASE1_CUSTOM_LIMIT" in os.environ:
                del os.environ["PHASE1_CUSTOM_LIMIT"]
                logger.info("Cleaned up PHASE1_CUSTOM_LIMIT environment variable")
    
    return results


def calculate_limit_statistics(accuracy_results: Dict[str, Dict[str, float]], limit: Optional[int]) -> Dict[str, float]:
    if not accuracy_results:
        return {
            'avg_accuracy': 0.0,
            'std_accuracy': 0.0,
            'min_accuracy': 0.0,
            'max_accuracy': 0.0,
            'num_model_task_pairs': 0,
            'num_successful_pairs': 0
        }
    
    all_accuracies = []
    for model_id, task_accuracies in accuracy_results.items():
        for task, accuracy in task_accuracies.items():
            if accuracy is not None:
                all_accuracies.append(accuracy)
    
    if not all_accuracies:
        return {
            'avg_accuracy': 0.0,
            'std_accuracy': 0.0,
            'min_accuracy': 0.0,
            'max_accuracy': 0.0,
            'num_model_task_pairs': 0,
            'num_successful_pairs': 0
        }
    
    return {
        'avg_accuracy': np.mean(all_accuracies),
        'std_accuracy': np.std(all_accuracies),
        'min_accuracy': np.min(all_accuracies),
        'max_accuracy': np.max(all_accuracies),
        'num_model_task_pairs': sum(len(task_acc) for task_acc in accuracy_results.values()),
        'num_successful_pairs': len(all_accuracies)
    }


def analyze_accuracy_convergence(results: Dict, exp_dir: Path) -> Tuple[Dict, str]:
    logger.info("Analyzing accuracy convergence")
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if len(valid_results) < 2:
        logger.error("Insufficient valid results for convergence analysis")
        return {}, "Insufficient data for analysis"
    
    sorted_limits = sorted(valid_results.keys(), key=lambda x: x if x is not None else float('inf'))
    
    convergence_analysis = {
        'convergence_point': None,
        'convergence_threshold': 0.05,
        'performance_data': [],
        'recommendations': []
    }
    
    for limit in sorted_limits:
        if limit in valid_results:
            stats = valid_results[limit]['statistics']
            convergence_analysis['performance_data'].append({
                'limit': limit,
                'avg_accuracy': stats['avg_accuracy'],
                'std_accuracy': stats['std_accuracy'],
                'num_successful_pairs': stats['num_successful_pairs']
            })
    
    perf_data = convergence_analysis['performance_data']
    convergence_point = None
    
    for i in range(1, len(perf_data)):
        current_acc = perf_data[i]['avg_accuracy']
        prev_acc = perf_data[i-1]['avg_accuracy']
        
        if prev_acc > 0:
            change_rate = abs(current_acc - prev_acc) / prev_acc
            if change_rate < convergence_analysis['convergence_threshold']:
                convergence_point = perf_data[i-1]['limit']
                logger.info(f"Convergence found at limit {convergence_point} (change rate: {change_rate:.3f})")
                break
    
    convergence_analysis['convergence_point'] = convergence_point
    
    recommendations = []
    
    if convergence_point is not None:
        recommendations.append(f"Accuracy converges at limit={convergence_point}")
        recommendations.append(f"Recommended limit for Phase 2: {convergence_point}")
        recommendations.append(f"Convergence threshold: {convergence_analysis['convergence_threshold']:.1%}")
        
        if convergence_point <= 50:
            recommendations.append("Low convergence point: Efficient execution with acceptable accuracy")
        elif convergence_point <= 200:
            recommendations.append("Moderate convergence point: Balanced speed-accuracy tradeoff")
        else:
            recommendations.append("High convergence point: Better accuracy but slower execution")
    else:
        best_limit = max(perf_data, key=lambda x: x['avg_accuracy'])['limit']
        recommendations.append(f"No convergence found within 5% threshold")
        recommendations.append(f"Using highest accuracy limit: {best_limit}")
        convergence_analysis['convergence_point'] = best_limit
    
    recommendations.append("Model-task combinations tested: " + str(sum(p['num_successful_pairs'] for p in perf_data)))
    recommendations.append("Accuracy range: " + 
                          f"{min(p['avg_accuracy'] for p in perf_data):.4f} - {max(p['avg_accuracy'] for p in perf_data):.4f}")
    
    convergence_analysis['recommendations'] = recommendations
    
    analysis_file = exp_dir / "analysis" / "accuracy_convergence_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(convergence_analysis, f, indent=2, default=str)
    
    report = generate_accuracy_convergence_report(convergence_analysis, perf_data, exp_dir)
    
    return convergence_analysis, report


def generate_accuracy_convergence_report(analysis: Dict, perf_data: List[Dict], exp_dir: Path) -> str:
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("PHASE 1 ACCURACY CONVERGENCE ANALYSIS REPORT")
    report_lines.append("Optimal Data Limit Determination for Model-Task Combinations")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("## EXPERIMENT OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"Limits tested: {len(perf_data)}")
    if perf_data:
        total_pairs = sum(p['num_successful_pairs'] for p in perf_data)
        report_lines.append(f"Total model-task pairs tested: {total_pairs}")
        report_lines.append(f"Convergence threshold: {analysis['convergence_threshold']:.1%}")
    report_lines.append("")
    
    report_lines.append("## ACCURACY PERFORMANCE BY LIMIT")
    report_lines.append("-" * 40)
    
    for data in perf_data:
        limit_str = f"Limit {data['limit']}" if data['limit'] is not None else "Full Dataset"
        report_lines.append(f"{limit_str:>15}: "
                          f"avg={data['avg_accuracy']:.4f}, "
                          f"std={data['std_accuracy']:.4f}, "
                          f"pairs={data['num_successful_pairs']}")
    
    report_lines.append("")
    
    report_lines.append("## CONVERGENCE ANALYSIS")
    report_lines.append("-" * 40)
    
    convergence_point = analysis['convergence_point']
    if convergence_point is not None:
        report_lines.append(f"Convergence point found: {convergence_point}")
        report_lines.append(f"Convergence threshold: {analysis['convergence_threshold']:.1%}")
        report_lines.append(f"Interpretation: Accuracy improvement < 5% beyond limit={convergence_point}")
    else:
        report_lines.append("No convergence point found within 5% threshold")
    
    report_lines.append("")
    
    report_lines.append("## RECOMMENDATIONS")
    report_lines.append("-" * 40)
    for rec in analysis['recommendations']:
        report_lines.append(f"- {rec}")
    report_lines.append("")
    
    report_lines.append("## NEXT STEPS")
    report_lines.append("-" * 40)
    
    if convergence_point is not None:
        report_lines.append(f"1. Use limit={convergence_point} for Phase 2 threshold optimization")
        report_lines.append("2. Proceed to Phase 2 with identified optimal limit")
    else:
        report_lines.append("1. Consider testing higher limits if accuracy is critical")
        report_lines.append("2. Proceed to Phase 2 with best performing limit")
    
    report_lines.append("3. Monitor accuracy vs execution time trade-offs")
    report_lines.append("4. Validate results with production workloads")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    report_file = exp_dir / "analysis" / "accuracy_convergence_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Accuracy convergence report saved: {report_file}")
    
    return report_text


def main():
    logger.info("Starting Phase 1: Accuracy Convergence Analysis")
    
    try:
        exp_dir = create_accuracy_experiment_dir()
        models = load_models_from_yaml()
        tasks = load_tasks_from_yaml()
        
        test_limits = [20, 40, 60, 80, 100, 120]
        
        config = {
            "experiment_type": "phase1_accuracy_convergence",
            "timestamp": datetime.now().isoformat(),
            "models": [{"id": m["id"], "name": m["name"]} for m in models],
            "tasks": tasks,
            "test_limits": test_limits,
            "convergence_threshold": 0.05,
            "total_combinations": len(models) * len(tasks)
        }
        
        config_file = exp_dir / "config" / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Phase 1 config: {len(models)} models × {len(tasks)} tasks = {config['total_combinations']} combinations")
        logger.info(f"Testing limits: {test_limits}")
        logger.info(f"Using 5% convergence threshold for analysis")
        
        results = run_accuracy_convergence_experiment(models, tasks, test_limits, exp_dir)
        
        results_file = exp_dir / "results" / "accuracy_convergence_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        analysis, report = analyze_accuracy_convergence(results, exp_dir)
        
        print("\n" + "=" * 80)
        print("PHASE 1 ACCURACY CONVERGENCE ANALYSIS COMPLETED")
        print("=" * 80)
        print(report)
        
        return {
            "success": True,
            "experiment_dir": exp_dir,
            "results": results,
            "analysis": analysis,
            "recommended_limit": analysis.get('convergence_point'),
            "convergence_found": analysis.get('convergence_point') is not None
        }
        
    except Exception as e:
        logger.error(f"Phase 1 accuracy convergence analysis failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    results = main()
    
    if results["success"]:
        print(f"\nPhase 1 Accuracy Convergence Analysis: SUCCESS!")
        print(f"Recommended limit: {results['recommended_limit']}")
        print(f"Convergence found: {results['convergence_found']}")
        print(f"Results saved in: {results['experiment_dir']}")
    else:
        print(f"\nPhase 1 Accuracy Convergence Analysis: FAILED!")
        print(f"Error: {results['error']}")