"""
Complete Fixed evaluation_lm.py
Dynamic model classification and enhanced thermal management
WandB removed for stability
Cooling logic removed for faster execution
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message="Exception ignored in.*TemporaryDirectory")
warnings.filterwarnings("ignore", message="Combined length of context.*exceeds model's maximum length")

import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download, HfApi
from lm_eval import evaluator
from code.config.config_loader import load_models, load_tasks
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import multiprocessing
import gc
import hashlib
import os
import signal
import time
import copy
import re
from typing import Dict, Tuple, Optional
import requests

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

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

multiprocessing.set_start_method('spawn', force=True)

FULL_RUN = False
ENABLE_TRACKING = True
TRACKING_MODE = "baseline"
performance_tracker = None
scheduler_manager = None
resource_monitor = None
config_manager = None

SCHEDULER_TIMEOUT = 30
TASK_TIMEOUT = 28800

EXPERIMENT_DIR = None

project_root = Path(__file__).parent.parent.parent
BASE_DIR = Path(__file__).parent
load_dotenv(project_root / ".env")

os.environ.setdefault("HF_HOME", str(project_root / "data" / "models"))
os.environ.setdefault("HF_DATASETS_CACHE", str(project_root / "data" / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
os.environ.setdefault("HF_EVALUATE_OFFLINE", "0")
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

gpu_list = [g.strip() for g in os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",") if g.strip()]
num_gpus = max(len(gpu_list), 1)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

models_config = getattr(sys.modules[__name__], 'models_config', load_models())
tasks = getattr(sys.modules[__name__], 'tasks', load_tasks().get("harness", []))


class DynamicModelClassifier:
    def __init__(self):
        self.size_patterns = [
            (r'(\d+\.\d+)\s*[-_]?\s*?b(?:illion)?(?:\b|[-_])', 'B'),
            (r'(\d+\.\d+)\s*m(?:illion)?(?:\b|[-_])', 'M'),
            (r'(\d+)\s*[-_]?\s*?b(?:illion)?(?:\b|[-_])', 'B'),
            (r'(\d+)[-_](\d+)b', 'SPECIAL'),
            (r'(\d+(?:\.\d+)?)\s*billion', 'B'),
            (r'(\d+(?:\.\d+)?)\s*million', 'M'),
        ]
        
        self.special_cases = {
            'seed-text': 1.5,
        }
        
        self.size_thresholds = {
            'small': (0, 3.0),
            'medium': (3.0, 15.0),  
            'large': (15.0, float('inf'))
        }
        
        self.thermal_properties = {
            'small': {'heat_factor': 1.0, 'cooling_time': 30},
            'medium': {'heat_factor': 2.5, 'cooling_time': 90},
            'large': {'heat_factor': 4.0, 'cooling_time': 180}
        }
        
        self.model_config_cache = {}
        self.hf_api = HfApi()
    
    def _fetch_model_info_from_api(self, model_id: str) -> Optional[float]:
        try:
            if model_id in self.model_config_cache:
                return self.model_config_cache[model_id]
            
            model_info = self.hf_api.model_info(model_id, timeout=10)
            
            if hasattr(model_info, 'config') and model_info.config:
                config = model_info.config
                
                if 'num_parameters' in config:
                    params = config['num_parameters']
                    size_b = params / 1e9
                    self.model_config_cache[model_id] = size_b
                    logger.info(f"API: {model_id} -> {size_b:.1f}B parameters")
                    return size_b
                
                if 'model_type' in config:
                    model_type = config['model_type'].lower()
                    
                    if 'llama' in model_type:
                        if '70b' in model_id.lower():
                            size_b = 70.0
                        elif '13b' in model_id.lower():
                            size_b = 13.0
                        elif '7b' in model_id.lower():
                            size_b = 7.0
                        else:
                            size_b = 7.0
                        
                        self.model_config_cache[model_id] = size_b
                        logger.info(f"API inference: {model_id} -> {size_b:.1f}B (llama)")
                        return size_b
            
            return None
            
        except Exception as e:
            logger.debug(f"API fetch failed for {model_id}: {e}")
            return None
    
    def _get_optimization_config(self, category: str, size_b: float) -> Dict:
        if category == 'large':
            return {
                'use_hflm': True,
                'load_in_8bit': True,
                'device_map': 'auto',
                'low_cpu_mem_usage': True,
                'offload_folder': './offload',
                'offload_state_dict': True,
                'default_batch_size': 1,
                'max_batch_size': 2,
                'use_gradient_checkpointing': True,
                'memory_efficient': True
            }
        elif category == 'medium':
            return {
                'use_hflm': False,
                'load_in_8bit': False,
                'device_map': 'auto' if size_b > 6 else None,
                'low_cpu_mem_usage': True,
                'default_batch_size': 2,
                'max_batch_size': 4,
                'memory_efficient': True
            }
        else:
            return {
                'use_hflm': False,
                'load_in_8bit': False,
                'device_map': None,
                'default_batch_size': 4,
                'max_batch_size': 16,
                'memory_efficient': False
            }
    
    def _extract_size_from_patterns(self, text: str) -> Optional[float]:
        text_clean = text.lower().replace('_', '-')

        logger.info(f"DEBUG: Trying to extract size from: '{text_clean}'") 
        
        for pattern, unit in self.size_patterns:
            matches = re.findall(pattern, text_clean, re.IGNORECASE)
            if matches:
                if unit == 'SPECIAL':
                    if len(matches[0]) == 2:
                        first, second = matches[0]
                        size_value = float(f"{first}.{second}")
                    else:
                        size_value = float(matches[0])
                elif unit == 'M':
                    size_value = float(matches[0]) / 1000.0
                else:
                    size_value = float(matches[0])
                
                logger.info(f"Pattern '{pattern}' matched -> {size_value}B")
                return size_value
            else:
                logger.info(f"Pattern '{pattern}' did not match")
        
        return None
    
    def extract_model_size(self, model_config: Dict) -> float:
        model_id = model_config.get("id", "")
        model_name = model_config.get("name", "")
        
        full_text = f"{model_id} {model_name}".lower()
        
        logger.info(f"Classifying model: {model_id}")
        
        pattern_size = self._extract_size_from_patterns(full_text)
        if pattern_size is not None:
            logger.info(f"Pattern match: {model_id} -> {pattern_size:.1f}B")
            return pattern_size
        
        api_size = self._fetch_model_info_from_api(model_id)
        if api_size is not None:
            logger.info(f"API match: {model_id} -> {api_size:.1f}B")
            return api_size
        
        for special_pattern, size in self.special_cases.items():
            if special_pattern in full_text:
                logger.info(f"Special case: {model_id} -> {size:.1f}B")
                return size
        
        logger.warning(f"No size found for {model_id}, using default 7.0B")
        return 7.0
    
    def classify_model(self, model_config: Dict) -> Tuple[str, float, Dict]:
        size_b = self.extract_model_size(model_config)
        
        if size_b < 3.0:
            category = 'small'
        elif size_b < 10.0:
            category = 'medium'
        else:
            category = 'large'
        
        config = self._get_optimization_config(category, size_b)
        
        logger.info(f"Model classified: {model_config.get('id')} -> {category} ({size_b:.1f}B)")
        return category, size_b, config

def get_model_classification_summary(models):
    classifier = DynamicModelClassifier()
    classification = {"large": [], "medium": [], "small": []}
    
    for idx, model in enumerate(models):
        category, size_b, _ = classifier.classify_model(model)
        classification[category].append((idx, model))
    
    logger.info("=" * 60)
    logger.info("MODEL CLASSIFICATION SUMMARY")
    logger.info("=" * 60)
    
    for category, models_list in classification.items():
        logger.info(f"{category.upper()} models ({len(models_list)}):")
        for idx, model in models_list:
            model_name = model.get("name", model.get("id", "").split("/")[-1])
            logger.info(f"  - {model_name}")
        logger.info("")
    
    return classification


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def platform_independent_timeout(func, timeout_seconds, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            logger.error(f"Operation timed out after {timeout_seconds} seconds")
            raise TimeoutError("Operation timeout")


def init_wandb_run(model_name, task_list, config_dict=None):
    logger.info("WandB disabled - skipping initialization")
    return None


def ensure_model_local(repo_id: str) -> str:
    cache_dir = os.environ["HF_HOME"]
    os.makedirs(cache_dir, exist_ok=True)
    local_path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=False)
    
    cfg_path = Path(local_path) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        max_pos = cfg.get("max_position_embeddings", 0)
        
        if max_pos > 1000000:
            logger.warning(f"Detected abnormal max_position_embeddings for {repo_id}: {max_pos}, resetting to default")
            if "gemma" in repo_id.lower():
                cfg["max_position_embeddings"] = 8192
            elif "llama" in repo_id.lower():
                cfg["max_position_embeddings"] = 4096
            else:
                cfg["max_position_embeddings"] = 2048
            cfg_path.write_text(json.dumps(cfg, indent=2))
            logger.info(f"Reset max_position_embeddings to {cfg['max_position_embeddings']}")
    
    if "gemma" in repo_id.lower():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        tokenizer_config_path = Path(local_path) / "tokenizer_config.json"
        
        if tokenizer.pad_token is None:
            if tokenizer_config_path.exists():
                tk_config = json.loads(tokenizer_config_path.read_text())
                tk_config["pad_token"] = tokenizer.eos_token
                tokenizer_config_path.write_text(json.dumps(tk_config, indent=2))
                logger.info(f"Set pad_token to eos_token for {repo_id}")
            
    return local_path


def run_evaluation(model, model_args, task_list, num_fewshot, batch_size, device, limit, gen_kwargs, extra_kwargs):
    eval_params = {
        "tasks": task_list,
        "num_fewshot": num_fewshot,
        "log_samples": num_fewshot < 5,
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
        return evaluator.simple_evaluate(model=model, model_args=model_args, **eval_params)
    else:
        return evaluator.simple_evaluate(model=model, **eval_params)


def extract_metrics_from_result(subtask_result):
    if not isinstance(subtask_result, dict):
        return {}
    
    metrics = {}
    
    metric_keys = {
        'accuracy': ['acc', 'accuracy', 'acc,none', 'accuracy,none'],
        'accuracy_norm': ['acc_norm', 'accuracy_norm', 'acc_norm,none', 'accuracy_norm,none'],
        'exact_match': ['exact_match', 'em', 'exact_match,none', 'em,none', 'exact_match,custom-extract'],
        'f1': ['f1', 'f1_score', 'f1,none', 'f1_score,none'],
        'bleu': ['bleu', 'bleu_score', 'bleu,none', 'bleu_score,none'],
        'rouge': ['rouge', 'rouge_score', 'rouge,none', 'rouge_score,none'],
        'perplexity': ['perplexity', 'ppl', 'perplexity,none', 'ppl,none'],
    }
    
    for metric_name, possible_keys in metric_keys.items():
        for key in possible_keys:
            if key in subtask_result:
                value = subtask_result[key]
                if value is not None and value != 'N/A':
                    metrics[metric_name] = value
                    break
    
    if not metrics:
        for key, value in subtask_result.items():
            if isinstance(value, (int, float)) and not key.startswith('_'):
                if 'stderr' not in key and 'std' not in key:
                    metrics[key] = value
    
    return metrics


def log_to_wandb(run, task_name, metrics, prefix="eval"):
    return


def clean_gpu_memory():
    try:
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
        time.sleep(2)
        
    except Exception as e:
        logger.warning(f"Error during GPU memory cleanup: {e}")


def evaluate_with_retry(model, model_args, task_list, initial_fewshot, batch_size, device, limit, gen_kwargs, extra_kwargs, run_name, wandb_run=None, tracker_context=None):
    all_results = {"results": {}}
    failed_tasks = []
    
    total_subtasks_count = 0
    tracking_records = {}
    
    zero_shot_tasks = ["gpqa", "hrm8k", "hrm-8k", "bbh", "agieval", "triviaqa", "nq_open", "nqopen", "humaneval", "csatqa"]
    
    for task_idx, task in enumerate(task_list):
        logger.info(f"{run_name}: Processing task {task_idx+1}/{len(task_list)}: {task}")
        
        clean_gpu_memory()
        
        if any(zs_task in task.lower() for zs_task in zero_shot_tasks):
            num_fewshot = 0
            logger.info(f"{run_name}: Task '{task}' detected as zero-shot task")
        else:
            num_fewshot = initial_fewshot
            logger.info(f"{run_name}: Task '{task}' will use num_fewshot={num_fewshot}")
        
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
                
                start_time = time.time()
                
                try:
                    result = platform_independent_timeout(
                        run_evaluation, 
                        TASK_TIMEOUT,
                        model, model_args, [task], num_fewshot, batch_size, device, limit, gen_kwargs, extra_kwargs
                    )
                except TimeoutError:
                    raise TimeoutError(f"Task {task} timed out")
                
                execution_time = time.time() - start_time
                
                if "results" in result:
                    task_results = result["results"]
                    if task_results:
                        all_results["results"].update(task_results)
                        task_completed = True
                        total_subtasks_count += len(task_results)
                        logger.info(f"{run_name}: Successfully completed task '{task}' with {len(task_results)} subtasks in {execution_time:.2f}s")
                        
                        if record_key and performance_tracker:
                            try:
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
                        
                        logger.info(f"\n{'='*60}")
                        logger.info(f"Task '{task}' Results:")
                        logger.info(f"{'='*60}")
                        
                        main_task_metrics = None
                        if task in task_results:
                            main_task_metrics = extract_metrics_from_result(task_results[task])
                        
                        for subtask_name, subtask_result in task_results.items():
                            metrics = extract_metrics_from_result(subtask_result)
                            
                            if metrics:
                                logger.info(f"  {subtask_name}:")
                                for metric_name, metric_value in metrics.items():
                                    if isinstance(metric_value, (int, float)):
                                        logger.info(f"    - {metric_name}: {metric_value:.4f}")
                                    else:
                                        logger.info(f"    - {metric_name}: {metric_value}")
                            else:
                                logger.info(f"  {subtask_name}: No standard metrics found")
                                logger.debug(f"    Raw result: {subtask_result}")
                        
                        logger.info(f"{'='*60}\n")
                    else:
                        logger.warning(f"{run_name}: Empty results for task '{task}'")
                        task_completed = True
                else:
                    logger.warning(f"{run_name}: No results key for task '{task}'")
                    task_completed = True
                    
            except TimeoutError:
                logger.error(f"{run_name}: Task '{task}' timed out after {TASK_TIMEOUT} seconds")
                failed_tasks.append(task)
                
                if record_key and performance_tracker:
                    try:
                        performance_tracker.record_end(
                            record_key=record_key,
                            status="failed",
                            error_message="Task timeout"
                        )
                    except:
                        pass
                break
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if "out of memory" in error_msg:
                    logger.error(f"{run_name}: CUDA OOM for task '{task}', skipping")
                    failed_tasks.append(task)
                    
                    if record_key and performance_tracker:
                        try:
                            performance_tracker.record_end(
                                record_key=record_key,
                                status="oom",
                                error_message=str(e)
                            )
                        except:
                            pass
                    
                    clean_gpu_memory()
                    break
                    
                elif "index out of range" in error_msg or "list index" in error_msg:
                    logger.info(f"{run_name}: IndexError for task '{task}' - {str(e)}")
                    if num_fewshot > 3:
                        logger.warning(f"{run_name}: Retrying task '{task}' with num_fewshot=3")
                        num_fewshot = 3
                    elif num_fewshot > 0:
                        logger.warning(f"{run_name}: Retrying task '{task}' with num_fewshot=0")
                        num_fewshot = 0
                    else:
                        logger.error(f"{run_name}: Task '{task}' failed with all num_fewshot values")
                        failed_tasks.append(task)
                        
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
                elif num_fewshot > 0 and ("fewshot" in error_msg or "exceeds the" in error_msg):
                    logger.warning(f"{run_name}: Fewshot error for task '{task}', retrying with num_fewshot=3")
                    num_fewshot = 3
                else:
                    logger.error(f"{run_name}: Task '{task}' failed with error: {e}")
                    failed_tasks.append(task)
                    
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
        
        clean_gpu_memory()
    
    successful_subtasks = list(all_results["results"].keys())
    logger.info(f"{run_name}: Completed evaluation - {len(successful_subtasks)} subtasks from {len(task_list)} requested tasks")
    if failed_tasks:
        logger.warning(f"{run_name}: Failed tasks: {failed_tasks}")
    
    return all_results


def setup_process_env():
    env_vars = {
        "HF_EVALUATE_OFFLINE": "0",
        "HF_ALLOW_CODE_EVAL": "1",
        "HF_HUB_OFFLINE": "0",
        "HF_DATASETS_OFFLINE": "0",
        "HF_HOME": str(project_root / "data" / "models"),
        "HF_DATASETS_CACHE": str(project_root / "data" / "datasets"),
        "TRANSFORMERS_VERBOSITY": "error",
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true"
    }
    os.environ.update(env_vars)
    
    if torch.cuda.is_available():
        torch.cuda.init()
    
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()]
        )
    
    for logger_name in ["transformers", "datasets", "huggingface_hub"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def initialize_tracking_components():
    global performance_tracker, scheduler_manager, resource_monitor, config_manager
    
    if not ENABLE_TRACKING or not TRACKER_AVAILABLE:
        return False
    
    try:
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(SCHEDULER_TIMEOUT)
        
        try:
            if CONFIG_MANAGER_AVAILABLE:
                config_manager = get_config(mode=TRACKING_MODE)
                logger.info(f"ConfigManager initialized for mode: {TRACKING_MODE}")
            
            if performance_tracker is None:
                performance_tracker = PerformanceTracker(mode=TRACKING_MODE)
                logger.info(f"PerformanceTracker initialized - mode: {TRACKING_MODE}")
            
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
                        config_manager=config_manager  
                    )
                    logger.info(f"SchedulerManager initialized - mode: {scheduler_manager.get_current_mode()}")
            
            return True
            
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        
    except TimeoutError:
        logger.error(f"Scheduler initialization timed out after {SCHEDULER_TIMEOUT} seconds")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize tracking components: {e}")
        return False


def evaluate_single(idx, mconf, task_list, full_run=False):
    setup_process_env()
    
    classifier = DynamicModelClassifier()
    category, size_b, opt_config = classifier.classify_model(mconf)
    
    model_id = mconf.get("id")
    model_name = mconf.get("name", model_id.split("/")[-1])
    run_name = f"{model_name}_harness_{idx+1}"
    gpu_id = gpu_list[idx % num_gpus]
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"[Process {os.getpid()}] {run_name} assigned to {device}")
    logger.info(f"Model: {model_name} | Size: {size_b:.1f}B | Category: {category.upper()}")

    is_exaone = "exaone" in model_id.lower()
    
    scheduler_batch_size = os.environ.get("SCHEDULER_BATCH_SIZE")
    scheduler_num_fewshot = os.environ.get("SCHEDULER_NUM_FEWSHOT")
    
    if scheduler_batch_size:
        batch_size = int(scheduler_batch_size)
        logger.info(f"Using scheduler recommended batch_size: {batch_size}")
    else:
        batch_size = opt_config['default_batch_size']
        logger.info(f"Using default batch_size for {category}: {batch_size}")

    custom_limit_env = os.environ.get("PHASE1_CUSTOM_LIMIT") or os.environ.get("PHASE2_CUSTOM_LIMIT")
    if custom_limit_env and custom_limit_env != "None":
        limit = int(custom_limit_env)
        logger.info(f"[Process {os.getpid()}] {run_name} - using custom limit: {limit}")
    elif full_run:
        limit = None
        logger.info(f"[Process {os.getpid()}] {run_name} - full_run: {full_run}, limit: None (full dataset)")
    else:
        limit = 2
        logger.info(f"[Process {os.getpid()}] {run_name} - full_run: {full_run}, limit: {limit} (test mode)")

    wandb_config = {
        "model_id": model_id,
        "model_name": model_name,
        "model_size_b": size_b,
        "model_category": category,
        "batch_size": batch_size,
        "device": device,
        "limit": limit,
        "tracking_mode": TRACKING_MODE,
        "scheduler_optimized": scheduler_batch_size is not None,
    }
    wandb_run = None

    try:
        local_path = ensure_model_local(model_id)
        config = AutoConfig.from_pretrained(local_path, local_files_only=True, trust_remote_code=is_exaone)
        max_seq = getattr(config, "max_position_embeddings", 2048)
        gen_tokens = min(max_seq - 128, 256)
        
        gen_kwargs = {"max_gen_toks": gen_tokens}
        
        if "gemma" in model_name.lower():
            gen_kwargs["max_gen_toks"] = min(gen_tokens, 256)
            logger.info(f"{run_name}: Gemma settings applied - max_gen_toks={gen_kwargs['max_gen_toks']}")
        
        code_tasks = ["humaneval", "mbpp", "apps"]
        has_code_task = any(task.lower() in code_tasks for task in task_list)
        
        if scheduler_num_fewshot:
            num_fewshot = int(scheduler_num_fewshot)
            logger.info(f"Using scheduler recommended num_fewshot: {num_fewshot}")
        elif not full_run:
            num_fewshot = 0
            logger.info(f"{run_name}: Test mode (limit=2), setting num_fewshot=0")
        else:
            num_fewshot = 5
        
        extra_kwargs = {}
        if has_code_task:
            extra_kwargs = {"write_out": True, "confirm_run_unsafe_code": True}
        
        tracker_context = {
            "model_id": model_id,
            "model_name": model_name,
            "model_size_b": size_b,
            "model_category": category,
            "gpu_id": gpu_id,
            "device": device,
            "batch_size": batch_size,
            "limit": limit,
            "tracking_mode": TRACKING_MODE,
            "scheduler_optimized": scheduler_batch_size is not None,
        }

        if opt_config['use_hflm']:
            from lm_eval.models.huggingface import HFLM
            
            clean_gpu_memory()
            
            model_kwargs = {
                "pretrained": local_path,
                "trust_remote_code": is_exaone,
                "device_map": opt_config['device_map'],
                "load_in_8bit": opt_config['load_in_8bit'],
                "low_cpu_mem_usage": opt_config['low_cpu_mem_usage'],
            }
            
            if opt_config.get('offload_folder'):
                model_kwargs.update({
                    "offload_folder": opt_config['offload_folder'],
                    "offload_state_dict": opt_config['offload_state_dict'],
                })
            
            if len(gpu_list) > 1 and not scheduler_batch_size:
                batch_size = min(batch_size * 2, opt_config['max_batch_size'])
            
            logger.info(f"{run_name}: Loading large model with 8bit={opt_config['load_in_8bit']}, batch_size={batch_size}")
            
            if "gemma" in model_name.lower():
                logger.info(f"{run_name}: Gemma model detected, will handle cache settings after loading")
            
            model = HFLM(**model_kwargs)
            results = evaluate_with_retry(model, None, task_list, num_fewshot, batch_size, device, limit, gen_kwargs, extra_kwargs, run_name, None, tracker_context)
        else:
            hf_args = f"pretrained={local_path}"
            if is_exaone:
                hf_args += ",trust_remote_code=True"
            
            if opt_config.get('device_map'):
                hf_args += f",device_map={opt_config['device_map']}"
            elif len(gpu_list) > 1:
                hf_args += ",device_map=auto"
                logger.info(f"{run_name}: Using device_map=auto for multi-GPU")
            
            if "gemma" in model_name.lower():
                logger.info(f"{run_name}: Gemma model detected, adjusting settings")
            
            results = evaluate_with_retry("hf", hf_args, task_list, num_fewshot, batch_size, device, limit, gen_kwargs, extra_kwargs, run_name, None, tracker_context)

        if EXPERIMENT_DIR:
            model_results_dir = EXPERIMENT_DIR / "model_results" / model_name
            model_results_dir.mkdir(parents=True, exist_ok=True)
            
            filename = model_results_dir / f"{run_name}.json"
            with open(filename, 'w') as f:
                json.dump(results.get("results", {}), f, indent=2, default=str)
            logger.info(f"[Process {os.getpid()}] Results saved to {filename}")
            
            legacy_results_dir = project_root / "results" / model_name
            legacy_results_dir.mkdir(parents=True, exist_ok=True)
            legacy_filename = legacy_results_dir / f"{run_name}.json"
            with open(legacy_filename, 'w') as f:
                json.dump(results.get("results", {}), f, indent=2, default=str)
            logger.info(f"[Process {os.getpid()}] Legacy results saved to {legacy_filename}")
        else:
            results_dir = project_root / "results" / model_name
            results_dir.mkdir(parents=True, exist_ok=True)
            
            filename = results_dir / f"{run_name}.json"
            with open(filename, 'w') as f:
                json.dump(results.get("results", {}), f, indent=2, default=str)
            logger.info(f"[Process {os.getpid()}] Results saved to {filename}")

        logger.info(f"[Process {os.getpid()}] Successfully completed {run_name}")
        
        clean_gpu_memory()
        
        return run_name, None

    except Exception as e:
        logger.error(f"[Process {os.getpid()}] Error in {run_name}: {e}", exc_info=True)
        
        clean_gpu_memory()
        
        return run_name, e


def optimize_schedule_for_thermal_management(schedule):
    if not schedule:
        return schedule
    
    classifier = DynamicModelClassifier()
    
    large_models = []
    medium_models = []
    small_models = []
    
    for task_priority in schedule:
        if hasattr(task_priority, 'estimated_memory'):
            memory_estimate = task_priority.estimated_memory
        else:
            memory_estimate = 20.0
        
        if memory_estimate > 30:
            large_models.append(task_priority)
        elif memory_estimate > 15:
            medium_models.append(task_priority)
        else:
            small_models.append(task_priority)
    
    optimized_schedule = []
    large_idx = medium_idx = small_idx = 0
    cooling_models_per_large = 2
    
    logger.info(f"Thermal optimization: {len(large_models)} large, {len(medium_models)} medium, {len(small_models)} small models")
    
    while large_idx < len(large_models) or medium_idx < len(medium_models) or small_idx < len(small_models):
        if large_idx < len(large_models):
            large_model = large_models[large_idx]
            optimized_schedule.append(large_model)
            large_idx += 1
            
            logger.info(f"Scheduled large model: {large_model.model_id}")
            
            cooling_count = 0
            while cooling_count < cooling_models_per_large:
                if small_idx < len(small_models):
                    cooling_model = small_models[small_idx]
                    optimized_schedule.append(cooling_model)
                    small_idx += 1
                    cooling_count += 1
                    logger.info(f"Scheduled cooling model: {cooling_model.model_id}")
                elif medium_idx < len(medium_models):
                    cooling_model = medium_models[medium_idx]
                    optimized_schedule.append(cooling_model)
                    medium_idx += 1
                    cooling_count += 1
                    logger.info(f"Scheduled cooling model: {cooling_model.model_id}")
                else:
                    break
        else:
            if medium_idx < len(medium_models):
                optimized_schedule.append(medium_models[medium_idx])
                medium_idx += 1
            elif small_idx < len(small_models):
                optimized_schedule.append(small_models[small_idx])
                small_idx += 1
    
    logger.info(f"Thermal optimization completed: {len(optimized_schedule)} total tasks scheduled")
    return optimized_schedule


def main():
    if EXPERIMENT_DIR:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using experiment directory: {EXPERIMENT_DIR}")
    
    (project_root / "results").mkdir(parents=True, exist_ok=True)
    
    global models_config, tasks, FULL_RUN, TRACKING_MODE, ENABLE_TRACKING
    global performance_tracker, scheduler_manager, resource_monitor, config_manager
    
    current_models_config = copy.deepcopy(models_config)
    current_tasks = copy.deepcopy(tasks)
    
    logger.info(f"Starting evaluation in {TRACKING_MODE} mode")
    logger.info(f"Models to process: {len(current_models_config)}")
    logger.info(f"Tasks to process: {len(current_tasks)}")
    
    try:
        tracking_initialized = initialize_tracking_components()
        
        if not tracking_initialized and ENABLE_TRACKING:
            logger.warning("Tracking components initialization failed, continuing without tracking")
    except Exception as e:
        logger.error(f"Failed to initialize tracking components: {e}")
        tracking_initialized = False
    
    logger.info("WandB logging disabled")
    
    if TRACKING_MODE == "optimized" and scheduler_manager:
        logger.info("="*60)
        logger.info("OPTIMIZED MODE - Using SchedulerManager")
        logger.info("="*60)
        
        try:
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(SCHEDULER_TIMEOUT)
            
            try:
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
                
                logger.info("Creating optimal schedule...")
                schedule = scheduler_manager.create_optimal_schedule(current_models_config, current_tasks)
                
                schedule = optimize_schedule_for_thermal_management(schedule)
                
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Optimal Execution Schedule ({current_mode.upper()} mode):")
            logger.info(f"{'='*80}")
            logger.info(f"{'Priority':<10} {'Model':<30} {'Task':<20} {'Est.Time':<10} {'Est.Mem':<10} {'GPU':<5}")
            logger.info(f"{'-'*80}")
            
            for i, task_priority in enumerate(schedule[:10]):
                model_name = next(
                    (m['name'] for m in current_models_config if m['id'] == task_priority.model_id),
                    task_priority.model_id.split("/")[-1]
                )
                logger.info(f"{task_priority.priority_score:>8.2f}   {model_name[:28]:<30} {task_priority.task_name[:18]:<20} "
                          f"{task_priority.estimated_time/3600:>6.1f}h   {task_priority.estimated_memory:>6.1f}GB  "
                          f"GPU{task_priority.suggested_gpu}")
            
            if len(schedule) > 10:
                logger.info(f"... and {len(schedule) - 10} more tasks")
            logger.info(f"{'='*80}\n")
            
            if EXPERIMENT_DIR:
                schedule_path = EXPERIMENT_DIR / "config" / f"schedule_{current_mode}.json"
            else:
                schedule_path = project_root / "results" / f"schedule_{current_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            scheduler_manager.export_schedule_with_metadata(schedule, schedule_path)
            logger.info(f"Schedule exported to: {schedule_path}")
            
            completed_tasks = []
            failed_tasks = []
            
            for task_idx, task_priority in enumerate(schedule):
                model_config = next(
                    (m for m in current_models_config if m['id'] == task_priority.model_id),
                    None
                )
                
                if not model_config:
                    logger.error(f"Model not found: {task_priority.model_id}")
                    failed_tasks.append((task_priority.model_id, task_priority.task_name))
                    continue
                
                model_idx = current_models_config.index(model_config)
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Executing task {task_idx+1}/{len(schedule)}")
                logger.info(f"Model: {model_config['name']}")
                logger.info(f"Task: {task_priority.task_name}")
                logger.info(f"Mode: {scheduler_manager.get_current_mode()}")
                logger.info(f"Suggested batch_size: {task_priority.suggested_batch_size}")
                logger.info(f"Suggested num_fewshot: {task_priority.suggested_num_fewshot}")
                logger.info(f"Rationale: {task_priority.rationale}")
                logger.info(f"{'='*60}")
                
                try:
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(SCHEDULER_TIMEOUT)
                    
                    next_task = scheduler_manager.get_next_task(schedule, completed_tasks)
                    
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                        
                    if next_task != task_priority:
                        logger.warning(f"SchedulerManager suggests different task, waiting...")
                        time.sleep(30)
                        continue
                        
                except TimeoutError:
                    logger.warning(f"Scheduler task selection timed out, proceeding with original schedule")
                except Exception as e:
                    logger.warning(f"Error in scheduler task selection: {e}, proceeding with original schedule")
                
                os.environ["SCHEDULER_BATCH_SIZE"] = str(task_priority.suggested_batch_size)
                os.environ["SCHEDULER_NUM_FEWSHOT"] = str(task_priority.suggested_num_fewshot)
                
                try:
                    run_name, error = evaluate_single(model_idx, model_config, [task_priority.task_name], FULL_RUN)
                    
                    if error:
                        logger.error(f"Task failed: {error}")
                        failed_tasks.append((task_priority.model_id, task_priority.task_name))
                        
                        try:
                            scheduler_manager.update_schedule_after_completion(
                                schedule[task_idx+1:],
                                task_priority,
                                0,
                                0,
                                "failed"
                            )
                        except Exception as update_error:
                            logger.warning(f"Error updating schedule after failure: {update_error}")
                    else:
                        logger.info(f"Task completed successfully")
                        completed_tasks.append((task_priority.model_id, task_priority.task_name))
                        
                        try:
                            scheduler_manager.update_schedule_after_completion(
                                schedule[task_idx+1:],
                                task_priority,
                                task_priority.estimated_time,
                                task_priority.estimated_memory,
                                "completed"
                            )
                        except Exception as update_error:
                            logger.warning(f"Error updating schedule after completion: {update_error}")
                
                finally:
                    if "SCHEDULER_BATCH_SIZE" in os.environ:
                        del os.environ["SCHEDULER_BATCH_SIZE"]
                    if "SCHEDULER_NUM_FEWSHOT" in os.environ:
                        del os.environ["SCHEDULER_NUM_FEWSHOT"]
                
                clean_gpu_memory()
            
            logger.info(f"\n{'='*60}")
            logger.info("Optimized Mode Execution Summary")
            logger.info(f"{'='*60}")
            logger.info(f"Total tasks scheduled: {len(schedule)}")
            logger.info(f"Successfully completed: {len(completed_tasks)}")
            logger.info(f"Failed tasks: {len(failed_tasks)}")
            
            try:
                final_stats = scheduler_manager.get_scheduler_statistics()
                logger.info(f"Final mode: {final_stats['current_mode']}")
                logger.info(f"Total training records: {final_stats['total_training_records']}")
                
                if final_stats['adaptive_status']['is_trained']:
                    logger.info(f"ML confidence: {final_stats['adaptive_status']['confidence']:.2f}")
            except Exception as e:
                logger.warning(f"Error getting final statistics: {e}")
            
            if failed_tasks:
                logger.info("\nFailed tasks:")
                for model_id, task_name in failed_tasks:
                    logger.info(f"  - {model_id} on {task_name}")
            
            logger.info(f"{'='*60}\n")
            
        except TimeoutError:
            logger.error("Scheduler operations timed out, falling back to baseline mode")
            TRACKING_MODE = "baseline"
        except Exception as e:
            logger.error(f"Error in optimized mode: {e}", exc_info=True)
            logger.info("Falling back to baseline mode...")
            TRACKING_MODE = "baseline"
            
    else:
        logger.info("="*60)
        logger.info("BASELINE MODE - Traditional Execution")
        logger.info("="*60)
        
        model_classification = get_model_classification_summary(current_models_config)
        
        large_models = model_classification["large"]
        medium_models = model_classification["medium"]
        small_models = model_classification["small"]
        
        total_models = len(large_models) + len(medium_models) + len(small_models)
        logger.info(f"Total models to process: {total_models}")
        logger.info(f"Large models: {len(large_models)}, Medium models: {len(medium_models)}, Small models: {len(small_models)}")
        
        all_models = [(idx, model) for idx, model in enumerate(current_models_config)]
        
        logger.info(f"Processing all {len(all_models)} models sequentially")
        
        for model_idx, (idx, model_config) in enumerate(all_models):
            model_name = model_config.get("name", model_config.get("id", "").split("/")[-1])
            
            classifier = DynamicModelClassifier()
            category, size_b, _ = classifier.classify_model(model_config)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing model {model_idx+1}/{len(all_models)}: {model_name} ({category}, {size_b:.1f}B)")
            logger.info(f"{'='*60}")
            
            try:
                run_name, error = evaluate_single(idx, model_config, current_tasks, FULL_RUN)
                if error:
                    logger.error(f"Run {run_name} failed: {error}")
                else:
                    logger.info(f"Run {run_name} finished successfully")
                    
            except Exception as e:
                logger.error(f"Unexpected error processing model {model_name}: {e}")
                continue
            
            clean_gpu_memory()
            time.sleep(3)
    
    if performance_tracker:
        try:
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
            
            if scheduler_manager:
                try:
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
                except Exception as e:
                    logger.warning(f"Error getting scheduler statistics: {e}")
            
            logger.info(f"{'='*60}\n")
            
            performance_tracker.close()
            if resource_monitor:
                resource_monitor.stop_monitoring()
                
        except Exception as e:
            logger.error(f"Error closing tracking components: {e}")


if __name__ == "__main__":
    main()