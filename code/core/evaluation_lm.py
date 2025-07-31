import os
# Prevent CUDA OOM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Handle temporary directory cleanup issues
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message="Exception ignored in.*TemporaryDirectory")
warnings.filterwarnings("ignore", message="Combined length of context.*exceeds model's maximum length")  # Ignore truncation warnings

import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download
from lm_eval.loggers import WandbLogger
from lm_eval import evaluator
from code.config.config_loader import load_models, load_tasks
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import gc
import wandb
import hashlib
import os

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# Performance Tracker and Scheduler imports
try:
    from code.scheduler.performance_tracker import PerformanceTracker
    from code.scheduler.scheduler_manager import SchedulerManager
    from code.scheduler.resource_monitor import ResourceMonitor
    from code.config.config_manager import get_config, ConfigManager
    TRACKER_AVAILABLE = True
    SCHEDULER_MANAGER_AVAILABLE = True
    CONFIG_MANAGER_AVAILABLE = True
except ImportError as e:
    TRACKER_AVAILABLE = False
    SCHEDULER_MANAGER_AVAILABLE = False
    CONFIG_MANAGER_AVAILABLE = False
    print(f"Warning: Scheduler components not available: {e}")

# Set multiprocessing to spawn mode (solves CUDA issues)
multiprocessing.set_start_method('spawn', force=True)

# Default FULL_RUN setting (can be overridden by entrance.py)
FULL_RUN = False

# Performance Tracker and Scheduler settings
ENABLE_TRACKING = True  # Enable/disable tracking
TRACKING_MODE = "baseline"  # baseline, optimized, or comparison
performance_tracker = None
scheduler_manager = None
resource_monitor = None
config_manager = None

# Global experiment directory (set by entrance.py)
EXPERIMENT_DIR = None

# Project root and environment variable loading - FIXED PATH
project_root = Path(__file__).parent.parent.parent
BASE_DIR = Path(__file__).parent
load_dotenv(project_root / ".env")

# WandB project/job type settings
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "lm-eval-harness-integration")
WANDB_JOB_TYPE = os.getenv("WANDB_JOB_TYPE", "eval")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")  # Optional: team/organization name

# Cache directory settings - FIXED PATHS
os.environ.setdefault("HF_HOME", str(project_root / "data" / "models"))
os.environ.setdefault("HF_DATASETS_CACHE", str(project_root / "data" / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
os.environ.setdefault("HF_EVALUATE_OFFLINE", "0")
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

# GPU settings (multi-GPU support)
gpu_list = [g.strip() for g in os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",") if g.strip()]
num_gpus = max(len(gpu_list), 1)

# Logging settings
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# WandB initialization helper
def init_wandb_run(model_name, task_list, config_dict=None):
    """Initialize WandB run"""
    # Check WandB API key
    if not os.getenv("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not found. WandB logging will be disabled.")
        return None
    
    # Generate run ID (use same run for restarts)
    run_id = hashlib.md5(f"{model_name}_{sorted(task_list)}".encode()).hexdigest()[:8]
    
    # Run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"
    
    # Default configuration
    default_config = {
        "model": model_name,
        "tasks": task_list,
        "num_tasks": len(task_list),
        "full_run": FULL_RUN,
        "num_gpus": num_gpus,
        "timestamp": timestamp,
    }
    
    if config_dict:
        default_config.update(config_dict)
    
    try:
        # Initialize WandB
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            job_type=WANDB_JOB_TYPE,
            name=run_name,
            id=run_id,
            config=default_config,
            resume="allow",  # Allow resuming interrupted runs
            reinit=True,  # Allow multiple runs in same process
        )
        
        logger.info(f"WandB run initialized: {run_name} (ID: {run_id})")
        return run
        
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        return None

# Model cache helper
def ensure_model_local(repo_id: str) -> str:
    cache_dir = os.environ["HF_HOME"]
    os.makedirs(cache_dir, exist_ok=True)
    local_path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=False)
    
    # Validate config.json for all models
    cfg_path = Path(local_path) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        max_pos = cfg.get("max_position_embeddings", 0)
        
        # Detect abnormally large values (over 1M)
        if max_pos > 1000000:
            logger.warning(f"Detected abnormal max_position_embeddings for {repo_id}: {max_pos}, resetting to default")
            # Set model-specific defaults
            if "gemma" in repo_id.lower():
                cfg["max_position_embeddings"] = 8192
            elif "llama" in repo_id.lower():
                cfg["max_position_embeddings"] = 4096
            else:
                cfg["max_position_embeddings"] = 2048  # Default value
            cfg_path.write_text(json.dumps(cfg, indent=2))
            logger.info(f"Reset max_position_embeddings to {cfg['max_position_embeddings']}")
    
    # Additional processing for Gemma models
    if "gemma" in repo_id.lower():
        # Check tokenizer and set pad_token
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        tokenizer_config_path = Path(local_path) / "tokenizer_config.json"
        
        if tokenizer.pad_token is None:
            # Set pad_token to eos_token
            if tokenizer_config_path.exists():
                tk_config = json.loads(tokenizer_config_path.read_text())
                tk_config["pad_token"] = tokenizer.eos_token
                tokenizer_config_path.write_text(json.dumps(tk_config, indent=2))
                logger.info(f"Set pad_token to eos_token for {repo_id}")
            
    return local_path

# Model and task loading
models_config = getattr(sys.modules[__name__], 'models_config', load_models())
tasks = getattr(sys.modules[__name__], 'tasks', load_tasks().get("harness", []))

# Evaluation execution helper function
def run_evaluation(model, model_args, task_list, num_fewshot, batch_size, device, limit, gen_kwargs, extra_kwargs):
    """Common function for running evaluation"""
    eval_params = {
        "tasks": task_list,
        "num_fewshot": num_fewshot,
        "log_samples": num_fewshot < 5,  # Enable log_samples when fewshot is reduced
        "rewrite_requests_cache": True,
        "cache_requests": False,
        "batch_size": batch_size,
        "use_cache": None,
        "device": device,
        "limit": limit,
        "gen_kwargs": gen_kwargs,
        **extra_kwargs,
    }
    
    if isinstance(model, str):
        # String case (regular model)
        return evaluator.simple_evaluate(model=model, model_args=model_args, **eval_params)
    else:
        # Object case (large model)
        return evaluator.simple_evaluate(model=model, **eval_params)

def extract_metrics_from_result(subtask_result):
    """Helper function to extract metrics from results"""
    if not isinstance(subtask_result, dict):
        return {}
    
    metrics = {}
    
    # Check all possible metric keys
    metric_keys = {
        'accuracy': ['acc', 'accuracy', 'acc,none', 'accuracy,none'],
        'accuracy_norm': ['acc_norm', 'accuracy_norm', 'acc_norm,none', 'accuracy_norm,none'],
        'exact_match': ['exact_match', 'em', 'exact_match,none', 'em,none', 'exact_match,custom-extract'],
        'f1': ['f1', 'f1_score', 'f1,none', 'f1_score,none'],
        'bleu': ['bleu', 'bleu_score', 'bleu,none', 'bleu_score,none'],
        'rouge': ['rouge', 'rouge_score', 'rouge,none', 'rouge_score,none'],
        'perplexity': ['perplexity', 'ppl', 'perplexity,none', 'ppl,none'],
    }
    
    # Check possible keys for each metric type
    for metric_name, possible_keys in metric_keys.items():
        for key in possible_keys:
            if key in subtask_result:
                value = subtask_result[key]
                if value is not None and value != 'N/A':
                    metrics[metric_name] = value
                    break
    
    # If no standard metrics found, find all numeric values
    if not metrics:
        for key, value in subtask_result.items():
            if isinstance(value, (int, float)) and not key.startswith('_'):
                # Exclude stderr and similar
                if 'stderr' not in key and 'std' not in key:
                    metrics[key] = value
    
    return metrics

def log_to_wandb(run, task_name, metrics, prefix="eval"):
    """Log metrics to WandB"""
    if not run:
        return
    
    try:
        # Convert metrics to WandB format
        wandb_metrics = {}
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                wandb_metrics[f"{prefix}/{task_name}/{metric_name}"] = metric_value
        
        # Log to WandB
        if wandb_metrics:
            run.log(wandb_metrics)
            
    except Exception as e:
        logger.warning(f"Failed to log metrics to WandB: {e}")

def evaluate_with_retry(model, model_args, task_list, initial_fewshot, batch_size, device, limit, gen_kwargs, extra_kwargs, run_name, wandb_run=None, tracker_context=None):
    """Evaluation function with automatic retry on errors - individual task execution"""
    all_results = {"results": {}}
    failed_tasks = []
    
    # Tasks that should start with zero-shot
    zero_shot_tasks = ["gpqa", "hrm8k", "hrm-8k", "bbh", "agieval", "triviaqa", "nq_open", "nqopen", "humaneval", "csatqa"]
    
    # Track overall progress
    total_subtasks_count = 0
    
    # Add tracking records dictionary
    tracking_records = {}
    
    # Execute each task individually
    for task_idx, task in enumerate(task_list):
        logger.info(f"{run_name}: Processing task {task_idx+1}/{len(task_list)}: {task}")
        
        # Log current task to WandB
        if wandb_run:
            wandb_run.log({
                "progress/current_task": task,
                "progress/task_index": task_idx + 1,
                "progress/total_tasks": len(task_list),
                "progress/percentage": (task_idx + 1) / len(task_list) * 100
            })
        
        # Determine num_fewshot per task
        if any(zs_task in task.lower() for zs_task in zero_shot_tasks):
            num_fewshot = 0
            logger.info(f"{run_name}: Task '{task}' detected as zero-shot task")
        else:
            num_fewshot = initial_fewshot
            logger.info(f"{run_name}: Task '{task}' will use num_fewshot={num_fewshot}")
        
        # Start Performance Tracker
        record_key = None
        if performance_tracker and tracker_context:
            try:
                tracking_config = {
                    **tracker_context,
                    "num_fewshot": num_fewshot,
                    "task_type": "harness",
                }
                record_key = performance_tracker.record_start(
                    model_id=tracker_context["model_id"],
                    model_name=tracker_context["model_name"],
                    task_name=task,
                    config=tracking_config
                )
                tracking_records[task] = record_key
            except Exception as e:
                logger.warning(f"Failed to start tracking for {task}: {e}")
        
        task_completed = False
        
        while num_fewshot >= 0 and not task_completed:
            try:
                logger.info(f"{run_name}: Evaluating task '{task}' with num_fewshot={num_fewshot}")
                result = run_evaluation(model, model_args, [task], num_fewshot, batch_size, device, limit, gen_kwargs, extra_kwargs)
                
                # Integrate results
                if "results" in result:
                    # Group tasks may have multiple subtask results
                    task_results = result["results"]
                    if task_results:  # If results are not empty
                        all_results["results"].update(task_results)
                        task_completed = True
                        total_subtasks_count += len(task_results)
                        logger.info(f"{run_name}: Successfully completed task '{task}' with {len(task_results)} subtasks")
                        
                        # Record success in Performance Tracker
                        if record_key and performance_tracker:
                            try:
                                # Extract main metrics
                                main_metrics = {}
                                if task in task_results:
                                    main_metrics = extract_metrics_from_result(task_results[task])
                                
                                performance_tracker.record_end(
                                    record_key=record_key,
                                    status="completed",
                                    results=main_metrics
                                )
                            except Exception as e:
                                logger.warning(f"Failed to record tracking end for {task}: {e}")
                        
                        # Output task scores and log to WandB
                        logger.info(f"\n{'='*60}")
                        logger.info(f"Task '{task}' Results:")
                        logger.info(f"{'='*60}")
                        
                        # Find main task metrics first
                        main_task_metrics = None
                        if task in task_results:
                            main_task_metrics = extract_metrics_from_result(task_results[task])
                        
                        for subtask_name, subtask_result in task_results.items():
                            # Extract metrics
                            metrics = extract_metrics_from_result(subtask_result)
                            
                            if metrics:
                                logger.info(f"  {subtask_name}:")
                                for metric_name, metric_value in metrics.items():
                                    if isinstance(metric_value, (int, float)):
                                        logger.info(f"    - {metric_name}: {metric_value:.4f}")
                                    else:
                                        logger.info(f"    - {metric_name}: {metric_value}")
                                
                                # Log to WandB
                                log_to_wandb(wandb_run, subtask_name, metrics)
                            else:
                                # If no metrics found, log entire result
                                logger.info(f"  {subtask_name}: No standard metrics found")
                                logger.debug(f"    Raw result: {subtask_result}")
                        
                        # Log main task summary to WandB
                        if main_task_metrics and wandb_run:
                            summary_metrics = {}
                            for metric_name, metric_value in main_task_metrics.items():
                                if isinstance(metric_value, (int, float)):
                                    summary_metrics[f"summary/{task}/{metric_name}"] = metric_value
                            if summary_metrics:
                                wandb_run.log(summary_metrics)
                        
                        logger.info(f"{'='*60}\n")
                    else:
                        logger.warning(f"{run_name}: Empty results for task '{task}'")
                        task_completed = True  # Treat empty results as completed
                else:
                    logger.warning(f"{run_name}: No results key for task '{task}'")
                    task_completed = True  # Treat no results as completed
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle CUDA OOM
                if "out of memory" in error_msg:
                    logger.error(f"{run_name}: CUDA OOM for task '{task}', skipping")
                    failed_tasks.append(task)
                    
                    # Record OOM tracking
                    if record_key and performance_tracker:
                        try:
                            performance_tracker.record_end(
                                record_key=record_key,
                                status="oom",
                                error_message=str(e)
                            )
                        except:
                            pass
                    
                    # Clean up memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                    
                # Handle IndexError
                elif "index out of range" in error_msg or "list index" in error_msg:
                    logger.info(f"{run_name}: IndexError for task '{task}' - {str(e)}")
                    if num_fewshot > 3:
                        logger.warning(f"{run_name}: Retrying task '{task}' with num_fewshot=3")
                        num_fewshot = 3
                    elif num_fewshot > 0:
                        logger.warning(f"{run_name}: Retrying task '{task}' with num_fewshot=0")
                        num_fewshot = 0
                    else:
                        # All attempts failed
                        logger.error(f"{run_name}: Task '{task}' failed with all num_fewshot values")
                        failed_tasks.append(task)
                        
                        # Record failure tracking
                        if record_key and performance_tracker:
                            try:
                                performance_tracker.record_end(
                                    record_key=record_key,
                                    status="failed",
                                    error_message=str(e)
                                )
                            except:
                                pass
                        break
                # Handle other fewshot-related errors
                elif num_fewshot > 0 and ("fewshot" in error_msg or "exceeds the" in error_msg):
                    logger.warning(f"{run_name}: Fewshot error for task '{task}', retrying with num_fewshot=3")
                    num_fewshot = 3
                else:
                    # Other errors are treated as task failure
                    logger.error(f"{run_name}: Task '{task}' failed with error: {e}")
                    failed_tasks.append(task)
                    
                    # Record failure tracking
                    if record_key and performance_tracker:
                        try:
                            performance_tracker.record_end(
                                record_key=record_key,
                                status="failed",
                                error_message=str(e)
                            )
                        except:
                            pass
                    break
        
        # Clean up memory (after each task)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Log result summary
    successful_subtasks = list(all_results["results"].keys())
    logger.info(f"{run_name}: Completed evaluation - {len(successful_subtasks)} subtasks from {len(task_list)} requested tasks")
    if failed_tasks:
        logger.warning(f"{run_name}: Failed tasks: {failed_tasks}")
    
    # Log final summary to WandB
    if wandb_run:
        wandb_run.log({
            "summary/total_subtasks": total_subtasks_count,
            "summary/successful_tasks": len(task_list) - len(failed_tasks),
            "summary/failed_tasks": len(failed_tasks),
            "summary/failed_task_names": ", ".join(failed_tasks) if failed_tasks else "None"
        })
    
    return all_results

# Environment setup helper
def setup_process_env():
    """Set environment variables in each process"""
    env_vars = {
        "HF_EVALUATE_OFFLINE": "0",
        "HF_ALLOW_CODE_EVAL": "1",
        "HF_HUB_OFFLINE": "0",
        "HF_DATASETS_OFFLINE": "0",
        "HF_HOME": str(project_root / "data" / "models"),
        "HF_DATASETS_CACHE": str(project_root / "data" / "datasets"),
        "TRANSFORMERS_VERBOSITY": "error"
    }
    os.environ.update(env_vars)
    
    # CUDA reinitialization
    if torch.cuda.is_available():
        torch.cuda.init()
    
    # Logging setup
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()]
        )
    
    # Adjust specific logger levels
    for logger_name in ["transformers", "datasets", "huggingface_hub"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
       

def initialize_tracking_components():
    global performance_tracker, scheduler_manager, resource_monitor, config_manager
    
    if not ENABLE_TRACKING or not TRACKER_AVAILABLE:
        return False
    
    try:
        # ConfigManager 초기화
        if CONFIG_MANAGER_AVAILABLE:
            config_manager = get_config()
            logger.info("ConfigManager initialized successfully")
        
        # PerformanceTracker 초기화
        if performance_tracker is None:
            performance_tracker = PerformanceTracker(mode=TRACKING_MODE)
            logger.info(f"PerformanceTracker initialized - mode: {TRACKING_MODE}")
        
        # Optimized 모드일 때만 SchedulerManager 초기화
        if TRACKING_MODE == "optimized" and SCHEDULER_MANAGER_AVAILABLE:
            if resource_monitor is None:
                resource_monitor = ResourceMonitor(
                    monitoring_interval=1.0,
                    history_size=300,
                    gpu_index=0
                )
                resource_monitor.start_monitoring()
                logger.info("ResourceMonitor initialized and started")
            
            if scheduler_manager is None:
                scheduler_manager = SchedulerManager(
                    performance_tracker=performance_tracker,
                    resource_monitor=resource_monitor,
                    num_gpus=num_gpus,
                    config_manager=config_manager  # ConfigManager 전달
                )
                logger.info(f"SchedulerManager initialized - mode: {scheduler_manager.get_current_mode()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize tracking components: {e}")
        return False
    
# Single model evaluation function
def evaluate_single(idx, mconf, task_list, full_run=False):
    setup_process_env()
    
    model_id = mconf.get("id")
    sid = model_id.split("/")[-1]
    run_name = f"{sid}_harness_{idx+1}"
    gpu_id = gpu_list[idx % num_gpus]
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Process {os.getpid()}] {run_name} assigned to {device}")

    # Check model type
    is_exaone = sid.startswith("EXAONE-3.5")
    is_large_model = any(x in sid.lower() for x in ["32b-instruct", "luxia", "12b", "13b", "30b", "70b"])
    is_medium_model = any(x in sid.lower() for x in ["7b", "8b"])
    
    # Check scheduler recommended settings
    scheduler_batch_size = os.environ.get("SCHEDULER_BATCH_SIZE")
    scheduler_num_fewshot = os.environ.get("SCHEDULER_NUM_FEWSHOT")
    
    # Determine batch size
    if scheduler_batch_size:
        # Use scheduler recommended value
        batch_size = int(scheduler_batch_size)
        logger.info(f"Using scheduler recommended batch_size: {batch_size}")
    else:
        # Existing logic
        if is_large_model:
            batch_size = 1  # Large models use batch size 1
        elif is_medium_model:
            batch_size = 1  # Medium models also reduced to 1
        else:
            batch_size = 1  # Small models also safely set to 1

    # Calculate sample limit - support custom limit from Phase 1/2
    custom_limit_env = os.environ.get("PHASE1_CUSTOM_LIMIT") or os.environ.get("PHASE2_CUSTOM_LIMIT")
    if custom_limit_env and custom_limit_env != "None":
        limit = int(custom_limit_env)
        logger.info(f"[Process {os.getpid()}] {run_name} - using custom limit: {limit}")
    elif full_run:
        limit = None
        logger.info(f"[Process {os.getpid()}] {run_name} - full_run: {full_run}, limit: None (full dataset)")
    else:
        limit = 2  # default for testing
        logger.info(f"[Process {os.getpid()}] {run_name} - full_run: {full_run}, limit: {limit} (test mode)")

    # Initialize WandB run
    wandb_config = {
        "model_id": model_id,
        "model_name": sid,
        "batch_size": batch_size,
        "is_large_model": is_large_model,
        "is_medium_model": is_medium_model,
        "device": device,
        "limit": limit,
        "tracking_mode": TRACKING_MODE,  # Add tracking mode
        "scheduler_optimized": scheduler_batch_size is not None,  # Whether scheduler is used
    }
    wandb_run = init_wandb_run(sid, task_list, wandb_config)

    try:
        local_path = ensure_model_local(model_id)
        config = AutoConfig.from_pretrained(local_path, local_files_only=True, trust_remote_code=is_exaone)
        max_seq = getattr(config, "max_position_embeddings", 2048)
        gen_tokens = min(max_seq - 128, 256)
        
        # Change generation_kwargs to gen_kwargs
        gen_kwargs = {"max_gen_toks": gen_tokens}
        
        # Additional safety measures for Gemma models
        if "gemma" in sid.lower():
            # More conservative generation length limit
            gen_kwargs["max_gen_toks"] = min(gen_tokens, 256)
            logger.info(f"{run_name}: Gemma settings applied - max_gen_toks={gen_kwargs['max_gen_toks']}")
        
        # Task-specific settings
        code_tasks = ["humaneval", "mbpp", "apps"]
        
        has_code_task = any(task.lower() in code_tasks for task in task_list)
        
        # Determine num_fewshot
        if scheduler_num_fewshot:
            # Use scheduler recommended value
            num_fewshot = int(scheduler_num_fewshot)
            logger.info(f"Using scheduler recommended num_fewshot: {num_fewshot}")
        elif not full_run:  # When limit=2 (test mode)
            num_fewshot = 0
            logger.info(f"{run_name}: Test mode (limit=2), setting num_fewshot=0")
        else:
            # Default is 5 (adjusted per task in evaluate_with_retry)
            num_fewshot = 5
        
        # Additional settings for code tasks
        extra_kwargs = {}
        if has_code_task:
            extra_kwargs = {"write_out": True, "confirm_run_unsafe_code": True}
        
        # Tracker context information
        tracker_context = {
            "model_id": model_id,
            "model_name": mconf.get("name", sid),
            "gpu_id": gpu_id,
            "device": device,
            "batch_size": batch_size,
            "limit": limit,
            "is_large_model": is_large_model,
            "is_medium_model": is_medium_model,
            "tracking_mode": TRACKING_MODE,
            "scheduler_optimized": scheduler_batch_size is not None,
        }

        # Large model processing
        if is_large_model:
            from lm_eval.models.huggingface import HFLM
            torch.cuda.empty_cache()
            
            # Decide 8bit quantization based on GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            logger.info(f"GPU memory available: {gpu_memory:.1f}GB")
            
            # Large models always use 8bit quantization
            use_8bit = True
            
            # Data parallel processing if multiple GPUs
            if len(gpu_list) > 1:
                model_kwargs = {
                    "pretrained": local_path,
                    "trust_remote_code": is_exaone,
                    "device_map": "auto",  # Automatically distribute across multiple GPUs
                    "load_in_8bit": use_8bit,
                    "low_cpu_mem_usage": True,
                }
                # Use default if no scheduler recommended batch size
                if not scheduler_batch_size:
                    batch_size = 4  # Batch size for large models
            else:
                model_kwargs = {
                    "pretrained": local_path,
                    "trust_remote_code": is_exaone,
                    "device_map": "auto",
                    "load_in_8bit": use_8bit,
                    "low_cpu_mem_usage": True,
                    "offload_folder": "./offload",
                    "offload_state_dict": True,
                }
            
            logger.info(f"{run_name}: Using 8bit={use_8bit}, batch_size={batch_size}")
            
            # Special handling for Gemma models
            if "gemma" in sid.lower():
                # Handle cache settings directly after model loading instead of config_kwargs
                logger.info(f"{run_name}: Gemma model detected, will handle cache settings after loading")
            
            model = HFLM(**model_kwargs)
            results = evaluate_with_retry(model, None, task_list, num_fewshot, batch_size, device, limit, gen_kwargs, extra_kwargs, run_name, wandb_run, tracker_context)
        else:
            # Regular model processing
            hf_args = f"pretrained={local_path}"
            if is_exaone:
                hf_args += ",trust_remote_code=True"
            
            # Use device_map for small models if multiple GPUs
            if len(gpu_list) > 1:
                hf_args += ",device_map=auto"
                logger.info(f"{run_name}: Using device_map=auto for multi-GPU")
            
            # Additional settings for Gemma models
            if "gemma" in sid.lower():
                logger.info(f"{run_name}: Gemma model detected, adjusting settings")
            
            results = evaluate_with_retry("hf", hf_args, task_list, num_fewshot, batch_size, device, limit, gen_kwargs, extra_kwargs, run_name, wandb_run, tracker_context)

        # Save results to experiment directory
        if EXPERIMENT_DIR:
            model_results_dir = EXPERIMENT_DIR / "model_results" / sid
            model_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to experiment directory
            filename = model_results_dir / f"{run_name}.json"
            with open(filename, 'w') as f:
                json.dump(results.get("results", {}), f, indent=2, default=str)
            logger.info(f"[Process {os.getpid()}] Results saved to {filename}")
            
            # Also save to legacy results directory for backward compatibility
            legacy_results_dir = project_root / "results" / sid
            legacy_results_dir.mkdir(parents=True, exist_ok=True)
            legacy_filename = legacy_results_dir / f"{run_name}.json"
            with open(legacy_filename, 'w') as f:
                json.dump(results.get("results", {}), f, indent=2, default=str)
            logger.info(f"[Process {os.getpid()}] Legacy results saved to {legacy_filename}")
        else:
            # Fallback to legacy results directory
            results_dir = project_root / "results" / sid
            results_dir.mkdir(parents=True, exist_ok=True)
            
            filename = results_dir / f"{run_name}.json"
            with open(filename, 'w') as f:
                json.dump(results.get("results", {}), f, indent=2, default=str)
            logger.info(f"[Process {os.getpid()}] Results saved to {filename}")

        # Save results file as WandB artifact
        if wandb_run:
            try:
                artifact = wandb.Artifact(
                    name=f"{sid}_results",
                    type="evaluation_results",
                    description=f"Evaluation results for {model_id}"
                )
                artifact.add_file(str(filename))
                wandb_run.log_artifact(artifact)
                logger.info(f"Results uploaded to WandB as artifact")
            except Exception as e:
                logger.warning(f"Failed to upload artifact to WandB: {e}")

        # WandB logging (using lm-eval's WandbLogger)
        if wandb_run:
            try:
                # Pass current run to lm-eval's WandbLogger
                wandb_logger = WandbLogger()
                
                # Set WandbLogger to use already initialized run
                if hasattr(wandb_logger, '_run'):
                    wandb_logger._run = wandb_run
                elif hasattr(wandb_logger, 'run'):
                    wandb_logger.run = wandb_run
                
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                
                if "samples" in results:
                    wandb_logger.log_eval_samples(results['samples'])
                    
                logger.info("Successfully logged to WandB using lm-eval WandbLogger")
            except Exception as wandb_err:
                logger.warning(f"lm-eval WandB logging failed, using direct logging: {wandb_err}")
                
                # Try direct logging
                if "results" in results:
                    final_metrics = {}
                    for task_name, task_result in results["results"].items():
                        metrics = extract_metrics_from_result(task_result)
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)):
                                final_metrics[f"final/{task_name}/{metric_value}"] = metric_value
                    
                    if final_metrics:
                        wandb_run.log(final_metrics)

        logger.info(f"[Process {os.getpid()}] Successfully completed {run_name}")
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Close WandB run
        if wandb_run:
            wandb_run.finish()
        
        return run_name, None

    except Exception as e:
        logger.error(f"[Process {os.getpid()}] Error in {run_name}: {e}", exc_info=True)
        
        # Close WandB run on error
        if wandb_run:
            wandb_run.finish(exit_code=1)
        
        return run_name, e

# Main evaluation function
def main():
    # Create experiment results directory
    if EXPERIMENT_DIR:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using experiment directory: {EXPERIMENT_DIR}")
    
    # Create legacy results directory for backward compatibility
    (project_root / "results").mkdir(parents=True, exist_ok=True)
    
    # Reference global variables
    global models_config, tasks, FULL_RUN, TRACKING_MODE, ENABLE_TRACKING
    global performance_tracker, scheduler_manager, resource_monitor, config_manager
    
    # Initialize tracking components
    tracking_initialized = initialize_tracking_components()
    
    if not tracking_initialized and ENABLE_TRACKING:
        logger.warning("Tracking components initialization failed, continuing without tracking")
    
    # Check WandB API key
    if not os.getenv("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set. WandB logging will be disabled.")
        logger.info("To enable WandB logging, set WANDB_API_KEY environment variable.")
    
    # Handle Optimized mode
    if TRACKING_MODE == "optimized" and scheduler_manager:
        logger.info("="*60)
        logger.info("OPTIMIZED MODE - Using SchedulerManager")
        logger.info("="*60)
        
        try:
            # Get current mode and statistics
            current_mode = scheduler_manager.get_current_mode()
            stats = scheduler_manager.get_scheduler_statistics()
            
            logger.info(f"Current Scheduler Mode: {current_mode}")
            logger.info(f"Training Records: {stats['total_training_records']}")
            logger.info(f"Configuration: min_learning_data={stats['configuration']['min_learning_data']}, "
                       f"stable_learning_data={stats['configuration']['stable_learning_data']}")
            
            if current_mode == "adaptive":
                logger.info(f"ML Confidence: {stats['adaptive_status']['confidence']:.2f}")
            elif current_mode == "hybrid":
                logger.info("Using Hybrid Mode (ML + Rule-based)")
            else:
                logger.info("Using Intelligent Mode (Rule-based)")
            
            # Create optimal schedule
            logger.info("Creating optimal schedule...")
            schedule = scheduler_manager.create_optimal_schedule(models_config, tasks)
            
            # Display schedule summary
            logger.info(f"\n{'='*80}")
            logger.info(f"Optimal Execution Schedule ({current_mode.upper()} mode):")
            logger.info(f"{'='*80}")
            logger.info(f"{'Priority':<10} {'Model':<30} {'Task':<20} {'Est.Time':<10} {'Est.Mem':<10} {'GPU':<5}")
            logger.info(f"{'-'*80}")
            
            for i, task_priority in enumerate(schedule[:10]):  # Show top 10 only
                model_name = next(
                    (m['name'] for m in models_config if m['id'] == task_priority.model_id),
                    task_priority.model_id.split("/")[-1]
                )
                logger.info(f"{task_priority.priority_score:>8.2f}   {model_name[:28]:<30} {task_priority.task_name[:18]:<20} "
                          f"{task_priority.estimated_time/3600:>6.1f}h   {task_priority.estimated_memory:>6.1f}GB  "
                          f"GPU{task_priority.suggested_gpu}")
            
            if len(schedule) > 10:
                logger.info(f"... and {len(schedule) - 10} more tasks")
            logger.info(f"{'='*80}\n")
            
            # Save schedule with metadata
            if EXPERIMENT_DIR:
                schedule_path = EXPERIMENT_DIR / "config" / f"schedule_{current_mode}.json"
            else:
                schedule_path = project_root / "results" / f"schedule_{current_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            scheduler_manager.export_schedule_with_metadata(schedule, schedule_path)
            logger.info(f"Schedule exported to: {schedule_path}")
            
            # Execute according to schedule
            completed_tasks = []
            failed_tasks = []
            
            for task_idx, task_priority in enumerate(schedule):
                # Find corresponding model
                model_config = next(
                    (m for m in models_config if m['id'] == task_priority.model_id),
                    None
                )
                
                if not model_config:
                    logger.error(f"Model not found: {task_priority.model_id}")
                    failed_tasks.append((task_priority.model_id, task_priority.task_name))
                    continue
                
                model_idx = models_config.index(model_config)
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Executing task {task_idx+1}/{len(schedule)}")
                logger.info(f"Model: {model_config['name']}")
                logger.info(f"Task: {task_priority.task_name}")
                logger.info(f"Mode: {scheduler_manager.get_current_mode()}")
                logger.info(f"Suggested batch_size: {task_priority.suggested_batch_size}")
                logger.info(f"Suggested num_fewshot: {task_priority.suggested_num_fewshot}")
                logger.info(f"Rationale: {task_priority.rationale}")
                logger.info(f"{'='*60}")
                
                # Check if we can proceed with next task
                next_task = scheduler_manager.get_next_task(schedule, completed_tasks)
                if next_task != task_priority:
                    logger.warning(f"SchedulerManager suggests different task, waiting...")
                    import time
                    time.sleep(30)
                    continue
                
                # Temporarily modify configuration to apply recommended settings
                original_models_config = models_config.copy()
                original_tasks = tasks.copy()
                
                # Set single model and task
                models_config = [model_config]
                tasks = [task_priority.task_name]
                
                # Pass scheduler recommendations via environment variables
                os.environ["SCHEDULER_BATCH_SIZE"] = str(task_priority.suggested_batch_size)
                os.environ["SCHEDULER_NUM_FEWSHOT"] = str(task_priority.suggested_num_fewshot)
                
                try:
                    # Execute single task
                    run_name, error = evaluate_single(model_idx, model_config, [task_priority.task_name], FULL_RUN)
                    
                    if error:
                        logger.error(f"Task failed: {error}")
                        failed_tasks.append((task_priority.model_id, task_priority.task_name))
                        
                        # Update schedule after failure
                        scheduler_manager.update_schedule_after_completion(
                            schedule[task_idx+1:],  # Remaining tasks
                            task_priority,
                            0,  # Execution time
                            0,  # Memory usage
                            "failed"
                        )
                    else:
                        logger.info(f"Task completed successfully")
                        completed_tasks.append((task_priority.model_id, task_priority.task_name))
                        
                        # Update schedule after successful completion
                        scheduler_manager.update_schedule_after_completion(
                            schedule[task_idx+1:],  # Remaining tasks
                            task_priority,
                            task_priority.estimated_time,  # Use predicted time
                            task_priority.estimated_memory,  # Use predicted memory
                            "completed"
                        )
                
                finally:
                    # Restore original settings
                    models_config = original_models_config
                    tasks = original_tasks
                    # Clean up environment variables
                    if "SCHEDULER_BATCH_SIZE" in os.environ:
                        del os.environ["SCHEDULER_BATCH_SIZE"]
                    if "SCHEDULER_NUM_FEWSHOT" in os.environ:
                        del os.environ["SCHEDULER_NUM_FEWSHOT"]
            
            # Execution completion summary
            logger.info(f"\n{'='*60}")
            logger.info("Optimized Mode Execution Summary")
            logger.info(f"{'='*60}")
            logger.info(f"Total tasks scheduled: {len(schedule)}")
            logger.info(f"Successfully completed: {len(completed_tasks)}")
            logger.info(f"Failed tasks: {len(failed_tasks)}")
            
            # Final scheduler statistics
            final_stats = scheduler_manager.get_scheduler_statistics()
            logger.info(f"Final mode: {final_stats['current_mode']}")
            logger.info(f"Total training records: {final_stats['total_training_records']}")
            
            if final_stats['adaptive_status']['is_trained']:
                logger.info(f"ML confidence: {final_stats['adaptive_status']['confidence']:.2f}")
            
            if failed_tasks:
                logger.info("\nFailed tasks:")
                for model_id, task_name in failed_tasks:
                    logger.info(f"  - {model_id} on {task_name}")
            
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"Error in optimized mode: {e}", exc_info=True)
            logger.info("Falling back to baseline mode...")
            # Fallback to Baseline mode
            TRACKING_MODE = "baseline"
            
    else:
        # Baseline mode (existing code)
        logger.info("="*60)
        logger.info("BASELINE MODE - Traditional Execution")
        logger.info("="*60)
        
        # Classify models by size
        large_models = []
        medium_models = []
        small_models = []
        
        for idx, mconf in enumerate(models_config):
            model_id = mconf.get("id")
            sid = model_id.split("/")[-1]
            is_large = any(x in sid.lower() for x in ["32b-instruct", "luxia", "12b", "13b", "30b", "70b"])
            is_medium = any(x in sid.lower() for x in ["7b", "8b"])
            
            if is_large:
                large_models.append((idx, mconf))
            elif is_medium:
                medium_models.append((idx, mconf))
            else:
                small_models.append((idx, mconf))
        
        # Strategy 1: Execute large models first (may need multiple GPUs)
        if large_models and len(gpu_list) > 1:
            logger.info(f"Processing {len(large_models)} large models first with all GPUs")
            for idx, mconf in large_models:
                # Large models execute sequentially, can use all GPUs
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
                run_name, error = evaluate_single(idx, mconf, tasks, FULL_RUN)
                if error:
                    logger.error(f"Run {run_name} failed: {error}")
                else:
                    logger.info(f"Run {run_name} finished successfully")
        
        # Strategy 2: Handle medium models (parallel if multiple GPUs, sequential otherwise)
        if medium_models:
            if len(gpu_list) > 1:
                logger.info(f"Processing {len(medium_models)} medium models in parallel")
                with ProcessPoolExecutor(max_workers=num_gpus) as executor:
                    futures = {
                        executor.submit(evaluate_single, idx, mconf, tasks, FULL_RUN): (idx, mconf) 
                        for idx, mconf in medium_models
                    }
                    for future in as_completed(futures):
                        run_name, error = future.result()
                        if error:
                            logger.error(f"Run {run_name} failed: {error}")
                        else:
                            logger.info(f"Run {run_name} finished successfully")
            else:
                logger.info(f"Processing {len(medium_models)} medium models sequentially")
                for idx, mconf in medium_models:
                    run_name, error = evaluate_single(idx, mconf, tasks, FULL_RUN)
                    if error:
                        logger.error(f"Run {run_name} failed: {error}")
                    else:
                        logger.info(f"Run {run_name} finished successfully")
        
        # Strategy 3: Small models execute in parallel
        if small_models:
            logger.info(f"Processing {len(small_models)} small models in parallel")
            with ProcessPoolExecutor(max_workers=num_gpus) as executor:
                futures = {
                    executor.submit(evaluate_single, idx, mconf, tasks, FULL_RUN): (idx, mconf) 
                    for idx, mconf in small_models
                }
                for future in as_completed(futures):
                    run_name, error = future.result()
                    if error:
                        logger.error(f"Run {run_name} failed: {error}")
                    else:
                        logger.info(f"Run {run_name} finished successfully")
        
        # Large models with single GPU
        if large_models and len(gpu_list) == 1:
            logger.info(f"Processing {len(large_models)} large models with single GPU")
            for idx, mconf in large_models:
                run_name, error = evaluate_single(idx, mconf, tasks, FULL_RUN)
                if error:
                    logger.error(f"Run {run_name} failed: {error}")
                else:
                    logger.info(f"Run {run_name} finished successfully")
    
    # Performance Tracker and Scheduler summary before program exit
    if performance_tracker:
        try:
            # Output overall statistics
            summary = performance_tracker.get_statistics_summary()
            logger.info(f"\n{'='*60}")
            logger.info("Performance Tracking Summary")
            logger.info(f"{'='*60}")
            logger.info(f"Mode: {TRACKING_MODE}")
            logger.info(f"Total runs: {summary['overall'].get('total_runs', 0)}")
            logger.info(f"Successful: {summary['overall'].get('successful_runs', 0)}")
            logger.info(f"OOM failures: {summary['overall'].get('oom_runs', 0)}")
            logger.info(f"Other failures: {summary['overall'].get('failed_runs', 0)}")
            total_time = summary['overall'].get('total_execution_time', 0)
            logger.info(f"Total execution time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
            
            # Scheduler Manager summary
            if scheduler_manager:
                scheduler_stats = scheduler_manager.get_scheduler_statistics()
                logger.info(f"\nScheduler Statistics:")
                logger.info(f"Final mode: {scheduler_stats['current_mode']}")
                logger.info(f"Mode transitions: {len(scheduler_stats['mode_transitions'])}")
                
                if scheduler_stats['adaptive_status']['is_trained']:
                    logger.info(f"ML model confidence: {scheduler_stats['adaptive_status']['confidence']:.2f}")
                    if 'model_performance' in scheduler_stats:
                        logger.info("ML model performance:")
                        for model_name, perf in scheduler_stats['model_performance'].items():
                            if 'accuracy' in perf:
                                logger.info(f"  {model_name}: {perf['accuracy']:.3f} accuracy")
                            elif 'mae' in perf:
                                logger.info(f"  {model_name}: {perf['mae']:.2f} MAE")
            
            logger.info(f"{'='*60}\n")
            
            # Close components
            performance_tracker.close()
            if resource_monitor:
                resource_monitor.stop_monitoring()
                
        except Exception as e:
            logger.error(f"Error closing tracking components: {e}")

if __name__ == "__main__":
    main()