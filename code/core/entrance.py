"""
Complete Fixed entrance.py
Enhanced mode independence and complete English localization
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import copy
import gc
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from code.config.config_loader import load_models, load_tasks
from code.core import evaluation_lm
import shutil
from dotenv import load_dotenv, set_key


def create_experiment_folder():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = project_root / "experiments_results" / f"exp_{timestamp}"
    
    subdirs = ["config", "model_results", "analysis_reports", "figures", "logs"]
    for subdir in subdirs:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"Created experiment folder: {experiment_dir}")
    return experiment_dir


def prompt_env_var(var_name):
    val = os.getenv(var_name)
    if not val:
        val = input(f"Please enter {var_name}: ").strip()
        if val:
            env_path = project_root / ".env"
            load_dotenv(env_path)
            set_key(env_path, var_name, val)
            os.environ[var_name] = val
    return os.getenv(var_name)


def print_separator():
    cols = shutil.get_terminal_size((80, 20)).columns
    print("=" * cols)


def display_menu(options, header):
    print_separator()
    print(header)
    print("0. Exit")
    print("Select all: type 'all' to select everything")
    for idx, opt in enumerate(options, start=1):
        print(f"{idx}. {opt}")
    print("\n(Enter numbers separated by space or comma, 0 to exit, 'all' to select all)")
    raw = input("Your selection: ").strip().lower()

    if raw == "all":
        return list(range(len(options)))

    choices = raw.replace(",", " ").split()
    if "0" in choices:
        print("Exiting program.")
        sys.exit(0)

    selected = []
    for c in choices:
        if c.isdigit():
            idx = int(c) - 1
            if 0 <= idx < len(options):
                selected.append(idx)
    return sorted(set(selected))


def complete_global_reset():
    print("Performing complete global state reset...")
    
    evaluation_lm.performance_tracker = None
    evaluation_lm.scheduler_manager = None
    evaluation_lm.resource_monitor = None
    evaluation_lm.config_manager = None
    
    env_vars_to_clear = [
        "SCHEDULER_BATCH_SIZE", "SCHEDULER_NUM_FEWSHOT", 
        "PHASE1_CUSTOM_LIMIT", "PHASE2_CUSTOM_LIMIT"
    ]
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
    
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    
    print("Global state reset completed")


def safe_execute_mode(mode, selected_models, selected_tasks, full_run, experiment_dir):
    print(f"\n=== Starting {mode.upper()} Mode Execution ===")
    
    complete_global_reset()
    
    independent_models = copy.deepcopy(selected_models)
    independent_tasks = copy.deepcopy(selected_tasks)
    
    evaluation_lm.TRACKING_MODE = mode
    evaluation_lm.models_config = independent_models
    evaluation_lm.tasks = independent_tasks
    evaluation_lm.FULL_RUN = full_run
    evaluation_lm.EXPERIMENT_DIR = experiment_dir
    evaluation_lm.ENABLE_TRACKING = True
    
    try:
        print(f"Executing {len(independent_models)} models with {len(independent_tasks)} tasks each")
        print(f"Expected total executions: {len(independent_models) * len(independent_tasks)}")
        
        evaluation_lm.main()
        
        print(f"{mode.upper()} mode execution completed successfully")
        return True
        
    except KeyboardInterrupt:
        print(f"\n{mode.upper()} mode execution interrupted by user")
        return False
    except MemoryError:
        print(f"Memory error in {mode} mode - consider reducing batch size")
        return False
    except Exception as e:
        print(f"Error in {mode} mode execution: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        complete_global_reset()
        print(f"{mode.upper()} mode cleanup completed")


def inter_mode_cleanup(delay_seconds: int = 15):
    print(f"\nWaiting {delay_seconds} seconds for complete system cleanup...")
    
    for i in range(delay_seconds):
        print(f"Cleanup progress: {i+1}/{delay_seconds}", end='\r')
        time.sleep(1)
    print("\nCleanup completed.")


def run_comparison_experiment(selected_models, selected_tasks, full_run, experiment_dir):
    print("\n=== Starting Comparison Experiment ===")
    print("Running Baseline and Optimized modes independently to compare performance.\n")
    
    total_expected = len(selected_models) * len(selected_tasks)
    print(f"Expected executions per mode: {total_expected}")
    print(f"Total expected executions: {total_expected * 2}")
    
    results = {"baseline": False, "optimized": False}
    
    print_separator()
    print("[1/2] Running Baseline mode...")
    print("Executing all models and tasks in traditional sequential order.")
    print_separator()
    
    baseline_success = safe_execute_mode("baseline", selected_models, selected_tasks, full_run, experiment_dir)
    results["baseline"] = baseline_success
    
    if baseline_success:
        print("Baseline mode completed successfully")
    else:
        print("Baseline mode failed")
    
    inter_mode_cleanup(15)
    complete_global_reset()
    
    print_separator()
    print("\n[2/2] Running Optimized mode...")
    print("Using intelligent scheduler to execute in optimized order.")
    print_separator()
    
    optimized_success = safe_execute_mode("optimized", selected_models, selected_tasks, full_run, experiment_dir)
    results["optimized"] = optimized_success
    
    if optimized_success:
        print("Optimized mode completed successfully")
    else:
        print("Optimized mode failed")
    
    print_separator()
    print("\n=== Comparison Experiment Summary ===")
    print(f"Baseline mode: {'SUCCESS' if results['baseline'] else 'FAILED'}")
    print(f"Optimized mode: {'SUCCESS' if results['optimized'] else 'FAILED'}")
    
    if results["baseline"] and results["optimized"]:
        print("Both modes completed successfully!")
        print("Execution results have been saved to Performance Tracker DB.")
        print("Use comparison_analyzer.py for detailed comparative analysis.")
    else:
        print("One or more modes failed. Check logs for details.")
        
    print_separator()
    
    return results


def main():
    env_path = project_root / ".env"
    load_dotenv(env_path)
    for key in ("OPENAI_API_KEY", "HF_API_TOKEN", "WANDB_API_KEY"):
        prompt_env_var(key)

    print_separator()
    print("Notice: Some models require separate access permissions from Hugging Face.")
    print("If you don't have access, please first generate a token at https://huggingface.co/settings/tokens")
    print("and request permission from the model's project page.")
    print("For more details, please refer to README.md.")
    print_separator()

    print_separator()
    print("KISTI SpectraBench Tool")
    print("\t\t\t\t2025. 06.18. SpectraBench version 1.0\n")
    print("\t\t\t\t(c) KISTI Large-scale AI Research Center All rights reserved.\n")
    print("\t\t\t\tThis tool is developed by AI Platform Team.\n")

    print_separator()

    models = load_models()
    harness_tasks = load_tasks().get("harness", [])

    model_names = [m.get("name", m.get("id")) for m in models]
    selected_model_idxs = display_menu(
        model_names,
        "Please select models to benchmark. Multiple selection is available."
    )
    selected_models = [models[i] for i in selected_model_idxs]

    task_names = harness_tasks
    selected_task_idxs = display_menu(
        task_names,
        "Please select benchmarks to run. Multiple selection is available."
    )
    selected_tasks = [harness_tasks[i] for i in selected_task_idxs]

    if not selected_models or not selected_tasks:
        print_separator()
        print("No models or benchmarks selected. Exiting program.")
        sys.exit(1)

    print_separator()
    print("Current benchmark is set to run with only 2 samples. If you want to use all benchmark data, please select below.")
    answer = input("Do you want to run full benchmark? (y/n): ").strip().lower()
    full_run = answer in ("", "y", "yes")
    print("Full benchmark mode: Running with limit=None\n") if full_run else print("Test mode: Running with limit=2\n")

    experiment_dir = create_experiment_folder()
    
    print_separator()
    print("EXPERIMENT CONFIGURATION")
    print_separator()
    print(f"Selected models ({len(selected_models)}):")
    for i, model in enumerate(selected_models, 1):
        print(f"  {i}. {model.get('name', model.get('id', ''))}")
    
    print(f"\nSelected tasks ({len(selected_tasks)}):")
    for i, task in enumerate(selected_tasks, 1):
        print(f"  {i}. {task}")
    
    total_executions = len(selected_models) * len(selected_tasks)
    print(f"\nTotal executions per mode: {total_executions}")
    print(f"Full run mode: {'Yes' if full_run else 'No (limit=2)'}")
    print_separator()

    print("Please select execution mode:")
    print("1. Baseline mode (Traditional approach)")
    print("   - Execute models and tasks sequentially")
    print("   - Use fixed settings as in traditional approach")
    print()
    print("2. Optimized mode (Intelligent scheduling)")
    print("   - Determine optimal execution order based on past execution data")
    print("   - Automatic batch size adjustment through memory usage prediction")
    print("   - Safe execution considering OOM risk")
    print()
    print("3. Comparison experiment mode (Baseline -> Optimized sequential execution)")
    print("   - Execute same models/tasks in both modes INDEPENDENTLY")
    print("   - Direct comparison of performance improvements")
    print("   - Each mode processes ALL selected models and tasks")
    print()
    
    mode_choice = input("Select (1-3): ").strip()
    
    if mode_choice == "1":
        print("\nRunning in Baseline mode.")        
        print_separator()
        print("\nStarting benchmark execution.")
        print(f"Selected models: {[m['name'] for m in selected_models]}")
        print(f"Selected benchmarks: {selected_tasks}\n")
        
        success = safe_execute_mode("baseline", selected_models, selected_tasks, full_run, experiment_dir)
        
        if success:
            print("Baseline mode execution completed successfully!")
        else:
            print("Baseline mode execution failed. Check logs for details.")
        
    elif mode_choice == "2":
        print("\nRunning in Optimized mode.")
        print("Intelligent scheduler will determine optimal execution order...")
        
        print_separator()
        print("\nStarting benchmark execution.")
        print(f"Selected models: {[m['name'] for m in selected_models]}")
        print(f"Selected benchmarks: {selected_tasks}\n")
        
        success = safe_execute_mode("optimized", selected_models, selected_tasks, full_run, experiment_dir)
        
        if success:
            print("Optimized mode execution completed successfully!")
        else:
            print("Optimized mode execution failed. Check logs for details.")
        
    elif mode_choice == "3":
        print("\nRunning in Comparison experiment mode.")
        print("Both modes will process ALL selected models and tasks independently.")
        
        confirm = input(f"\nThis will run {total_executions} executions in each mode (total: {total_executions * 2}). Continue? (y/n): ").strip().lower()
        if confirm not in ("", "y", "yes"):
            print("Experiment cancelled.")
            sys.exit(0)
        
        results = run_comparison_experiment(selected_models, selected_tasks, full_run, experiment_dir)
        
        if results["baseline"] and results["optimized"]:
            print("Comparison experiment completed successfully!")
        else:
            print("Comparison experiment partially failed. Check individual mode results.")
        
    else:
        print("Invalid selection. Exiting program.")
        sys.exit(1)


if __name__ == "__main__":
    main()