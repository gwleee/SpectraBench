import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from code.config.config_loader import load_models, load_tasks
from code.core import evaluation_lm
import shutil
from dotenv import load_dotenv, set_key


def create_experiment_folder():
    """Create timestamped experiment folder"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = project_root / "experiments_results" / f"exp_{timestamp}"
    
    # Create all necessary subdirectories
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
            # Write to .env file
            env_path = project_root / ".env"
            # Load and set
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
    print("Select all: all or 전체")
    for idx, opt in enumerate(options, start=1):
        print(f"{idx}. {opt}")
    print("\n(Enter numbers separated by space or comma, 0 to exit, 'all' to select all)")
    raw = input("User input: ").strip().lower()

    if raw in ("all", "전체"):
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


def run_comparison_experiment(selected_models, selected_tasks, full_run, experiment_dir):
    """Run Baseline and Optimized sequentially for comparison"""
    print("\n=== Starting Comparison Experiment ===")
    print("Running Baseline and Optimized modes sequentially to compare performance.\n")
    
    # 1. Baseline execution
    print_separator()
    print("[1/2] Running Baseline mode...")
    print("Executing models and tasks sequentially in traditional order.")
    print_separator()
    
    evaluation_lm.TRACKING_MODE = "baseline"
    evaluation_lm.models_config = selected_models
    evaluation_lm.tasks = selected_tasks
    evaluation_lm.FULL_RUN = full_run
    evaluation_lm.EXPERIMENT_DIR = experiment_dir
    evaluation_lm.main()
    
    # 2. Optimized execution
    print_separator()
    print("\n[2/2] Running Optimized mode...")
    print("Using intelligent scheduler to execute in optimized order.")
    print_separator()
    
    evaluation_lm.TRACKING_MODE = "optimized"
    evaluation_lm.models_config = selected_models
    evaluation_lm.tasks = selected_tasks
    evaluation_lm.FULL_RUN = full_run
    evaluation_lm.EXPERIMENT_DIR = experiment_dir
    evaluation_lm.main()
    
    # 3. Generate comparison report
    print_separator()
    print("\n=== Comparison Experiment Complete ===")
    print("Execution results have been saved to Performance Tracker DB.")
    print("Use comparison_analyzer.py for detailed comparative analysis.")
    print_separator()


def main():
    # Load .env and check keys
    env_path = project_root / ".env"
    load_dotenv(env_path)
    for key in ("OPENAI_API_KEY", "HF_API_TOKEN", "WANDB_API_KEY"):
        prompt_env_var(key)

    # Model permission notice
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

    # Create experiment folder
    experiment_dir = create_experiment_folder()

    # Execution mode selection
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
    print("3. Comparison experiment mode (Baseline → Optimized sequential execution)")
    print("   - Execute same models/tasks in both modes")
    print("   - Direct comparison of performance improvements")
    print()
    
    mode_choice = input("Select (1-3): ").strip()
    
    if mode_choice == "1":
        evaluation_lm.TRACKING_MODE = "baseline"
        evaluation_lm.ENABLE_TRACKING = True
        print("\nRunning in Baseline mode.")
        
        print_separator()
        print("\nStarting benchmark execution.")
        print(f"Selected models: {[m['name'] for m in selected_models]}")
        print(f"Selected benchmarks: {selected_tasks}\n")
        
        evaluation_lm.models_config = selected_models
        evaluation_lm.tasks = selected_tasks
        evaluation_lm.FULL_RUN = full_run
        evaluation_lm.EXPERIMENT_DIR = experiment_dir
        evaluation_lm.main()
        
    elif mode_choice == "2":
        evaluation_lm.TRACKING_MODE = "optimized"
        evaluation_lm.ENABLE_TRACKING = True
        print("\nRunning in Optimized mode.")
        print("Intelligent scheduler will determine optimal execution order...")
        
        print_separator()
        print("\nStarting benchmark execution.")
        print(f"Selected models: {[m['name'] for m in selected_models]}")
        print(f"Selected benchmarks: {selected_tasks}\n")
        
        evaluation_lm.models_config = selected_models
        evaluation_lm.tasks = selected_tasks
        evaluation_lm.FULL_RUN = full_run
        evaluation_lm.EXPERIMENT_DIR = experiment_dir
        evaluation_lm.main()
        
    elif mode_choice == "3":
        evaluation_lm.ENABLE_TRACKING = True
        print("\nRunning in Comparison experiment mode.")
        run_comparison_experiment(selected_models, selected_tasks, full_run, experiment_dir)
        
    else:
        print("Invalid selection. Exiting program.")
        sys.exit(1)

if __name__ == "__main__":
    main()