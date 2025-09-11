# persistence.py
import json, os
import torch
from typing import Dict, Any

def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_model_state_dict(path: str, state_dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
