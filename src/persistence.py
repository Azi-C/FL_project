# persistence.py
import json
import os
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


# ---------------- CSV LOGGING HELPERS ----------------
import csv
from pathlib import Path

def ensure_csv(path: str, fieldnames: list[str]) -> None:
    """Create CSV with header if it doesn't exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

def append_csv_row(path: str, row: dict, fieldnames: list[str]) -> None:
    """Append one row to CSV (assumes ensure_csv was called)."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)
