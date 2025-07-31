import yaml, importlib
from pathlib import Path

BASE = Path(__file__).parent

def load_models():
    data = yaml.safe_load((BASE / "models.yaml").read_text())
    models = data.get("models", [])
    for m in models:
        m.setdefault("model_args", {})
    return models

def load_tasks():
    data = yaml.safe_load((BASE / "tasks.yaml").read_text())
    return {
        "harness": data.get("harness_tasks", []),
        "custom":  data.get("custom_tasks", [])
    }

def load_custom_tasks():
    data = yaml.safe_load((BASE/"tasks.yaml").read_text())
    custom = data.get("custom_tasks", [])
    instances = []
    for c in custom:
        mod = importlib.import_module(c["module"])
        TaskClass = getattr(mod, "Task")
        instances.append(TaskClass(few_shot=c.get("few_shot", 0),
                                   input_format=c.get("input_format", None)))
    return instances