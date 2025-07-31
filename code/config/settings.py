from dotenv import load_dotenv
import os
from pathlib import Path
from config.config_loader import load_models, load_tasks
from config.config_loader import load_custom_tasks

BASE_DIR = Path(__file__).parent.parent

load_dotenv(BASE_DIR / ".env")

os.environ.setdefault("TRANSFORMERS_CACHE", str(BASE_DIR / "data" / "models"))
os.environ.setdefault("HF_DATASETS_CACHE", str(BASE_DIR / "data" / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MODELS        = load_models()                # [{name,id,type,loader},…]
TASKS         = load_tasks()["harness"]      # ["hellaswag",…]
CUSTOM_TASKS  = load_tasks()["custom"] or [] # [{name,git_url,…},…] or []

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_TOKEN   = os.getenv("HF_API_TOKEN")
WANDB_API_KEY  = os.getenv("WANDB_API_KEY")
DEVICE         = os.getenv("DEVICE", "cuda")

MC_TASKS = {
    "kmmlu", "haerae", "kobest", "belebele",
    "csatqa", "hrm8k", "kormedmcqa",
    "mmlu", "mmlu_pro", "bbh",
    "arc_challenge", "arc_easy", "hellaswag",
    "gpqa", "piqa", "cstqa", "gsm8k"
}