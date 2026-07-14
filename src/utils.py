"""
Utility helpers used across every module:
  - reproducibility seeding
  - consistent logging format
  - model save / load wrappers
  - metrics pretty-printing
"""

import os
import random
import pickle
import json
import logging
import numpy as np
import torch
import joblib

from src.config import SEED, MODEL_DIR, OUTPUT_DIR


def set_seed(seed: int = SEED):
    """Pin every random source so experiments are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to stdout with a clean format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(message)s",
                                               datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def save_sklearn_model(model, filename: str):
    """Persist a scikit-learn model to the models directory."""
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    return path


def load_sklearn_model(filename: str):
    """Load a scikit-learn model from the models directory."""
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path)


def save_torch_model(model: torch.nn.Module, filename: str, extra: dict = None):
    """Save PyTorch model weights and optional metadata dict."""
    path = os.path.join(MODEL_DIR, filename)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_torch_model(model: torch.nn.Module, filename: str) -> torch.nn.Module:
    """Load weights into a pre-built model instance."""
    path = os.path.join(MODEL_DIR, filename)
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    return model


def save_metrics(metrics: dict, filename: str):
    """Write a metrics dictionary to JSON in the outputs directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def print_metrics(metrics: dict, title: str = "Results"):
    """Pretty-print a standard metrics dict."""
    print(f"\n{title}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    if "inference_ms" in metrics:
        print(f"  Inference: {metrics['inference_ms']:.4f} ms/sample")
