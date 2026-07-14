"""
Tier 3b — Feature Boosting Continual Learning (FBCL).

Trains incrementally on fault severity groups (tasks):
  Task 0: all classes (initial training)
  Task 1: new severity subset introduced — adapter + distillation
  Task 2: further new classes — adapter + distillation

Knowledge distillation from the frozen old model prevents catastrophic
forgetting of previously learned fault types.

Run standalone:
    python src/train_continual.py
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SEED, DEVICE, N_CLASSES,
    FBCL_EPOCHS_PER_TASK, FBCL_LR, FBCL_DISTILL_WEIGHT, FBCL_MEMORY_SIZE,
    DL_BATCH_SIZE, DL_GRAD_CLIP,
)
from src.models import FBCLModel
from src.utils import set_seed, get_logger, save_torch_model, save_metrics
from src.artifacts import build_artifact_registry
from src.preprocessing import load_preprocessed

log = get_logger("train_continual")


# Define incremental tasks as subsets of the 10 classes.
# Task 0 = initial 4 classes, Task 1 adds 3 more, Task 2 finishes the rest.
TASK_CLASSES = [
    [0, 1, 2, 6],        # Task 0: Ball faults + Normal
    [3, 4, 5],           # Task 1: Inner race faults
    [7, 8, 9],           # Task 2: Outer race faults
]


def get_task_data(X, y, class_list):
    """Filter dataset to only the given list of class indices."""
    mask = np.isin(y, class_list)
    # Remap labels to 0..len(class_list)-1 for the task head
    remap = {c: i for i, c in enumerate(class_list)}
    y_task = np.array([remap[yi] for yi in y[mask]])
    return X[mask], y_task


def make_memory_buffer(X, y, memory_size: int):
    """
    Keep `memory_size` exemplars per class (reservoir sampling strategy).
    Used to replay past data during new task training.
    """
    mem_X, mem_y = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        chosen = np.random.choice(idx, size=min(memory_size, len(idx)), replace=False)
        mem_X.append(X[chosen])
        mem_y.append(y[chosen])
    return np.vstack(mem_X), np.concatenate(mem_y)


def distillation_loss(new_logits, old_logits, temperature: float = 2.0):
    """
    Soft knowledge distillation loss.
    Encourages the new model to preserve old predictions at higher temperature.
    """
    p_old = F.softmax(old_logits / temperature, dim=1)
    p_new = F.log_softmax(new_logits / temperature, dim=1)
    return F.kl_div(p_new, p_old, reduction="batchmean") * (temperature ** 2)


def train_task(model: FBCLModel, task_id: int,
               X_task, y_task, old_model,
               memory_X=None, memory_y=None):
    """
    Train the model on one task. If task_id > 0, mixes in memory replay and
    distillation from the frozen old_model to prevent forgetting.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=FBCL_LR, weight_decay=3e-4)

    # Combine current task data with memory buffer if available
    if memory_X is not None and len(memory_X) > 0:
        X_all = np.vstack([X_task, memory_X])
        y_all = np.concatenate([y_task, memory_y])
    else:
        X_all, y_all = X_task, y_task

    ds = TensorDataset(torch.tensor(X_all, dtype=torch.float32),
                       torch.tensor(y_all, dtype=torch.long))
    loader = DataLoader(ds, batch_size=DL_BATCH_SIZE, shuffle=True, num_workers=0)

    log.info(f"Task {task_id}: {len(X_task)} new samples  "
             f"({len(memory_X) if memory_X is not None else 0} memory)")

    for epoch in range(1, FBCL_EPOCHS_PER_TASK + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            # Classification loss on current task head
            logits = model(xb, task_id=task_id)
            cls_loss = F.cross_entropy(logits, yb)

            # Distillation from old model on current batch (prevents forgetting)
            dist_loss = torch.tensor(0.0, device=DEVICE)
            if old_model is not None and task_id > 0:
                with torch.no_grad():
                    old_logits = old_model(xb, task_id=task_id - 1)
                new_logits_old_head = model(xb, task_id=task_id - 1)
                dist_loss = distillation_loss(new_logits_old_head, old_logits)

            loss = cls_loss + FBCL_DISTILL_WEIGHT * dist_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), DL_GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            log.info(f"  Task {task_id}  Epoch {epoch}  loss={total_loss/len(loader):.4f}")

    return model


def train_fbcl():
    set_seed(SEED)
    log.info("Loading preprocessed data")
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = load_preprocessed()
    n_feat = X_train.shape[1]

    # Start with initial task classes
    n_initial = len(TASK_CLASSES[0])
    model     = FBCLModel(n_feat, initial_classes=n_initial).to(DEVICE)
    old_model = None
    memory_X  = np.zeros((0, n_feat), dtype=np.float32)
    memory_y  = np.zeros(0, dtype=int)

    task_results = []

    for task_id, class_list in enumerate(TASK_CLASSES):
        if task_id > 0:
            model.add_task(n_new_classes=len(class_list))
            model = model.to(DEVICE)

        X_task, y_task = get_task_data(X_train, y_train, class_list)

        # Keep a snapshot of the old model for distillation
        old_model = copy.deepcopy(model) if task_id > 0 else None
        if old_model:
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False

        model = train_task(model, task_id, X_task, y_task, old_model,
                           memory_X, memory_y)

        # Update memory buffer with exemplars from this task
        new_mem_X, new_mem_y = make_memory_buffer(X_task, y_task, FBCL_MEMORY_SIZE)
        memory_X = np.vstack([memory_X, new_mem_X]) if len(memory_X) > 0 else new_mem_X
        memory_y = np.concatenate([memory_y, new_mem_y])

        # Evaluate accuracy on this task's test split
        X_te_t, y_te_t = get_task_data(X_test, y_test, class_list)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_te_t, dtype=torch.float32).to(DEVICE),
                          task_id=task_id).argmax(1).cpu().numpy()
        acc = accuracy_score(y_te_t, preds)
        task_results.append({"task_id": task_id, "classes": class_list, "accuracy": float(acc)})
        log.info(f"Task {task_id} test accuracy: {acc:.4f}")

    # Average accuracy across tasks (continual learning main metric)
    avg_acc = float(np.mean([r["accuracy"] for r in task_results]))
    log.info(f"FBCL average task accuracy: {avg_acc:.4f}")

    save_torch_model(model, "fbcl_model.pth",
                     extra={"n_features": n_feat, "task_classes": TASK_CLASSES})
    metrics = {"task_results": task_results, "average_accuracy": avg_acc}
    save_metrics(metrics, "fbcl_metrics.json")
    build_artifact_registry()
    return model, metrics


if __name__ == "__main__":
    train_fbcl()
