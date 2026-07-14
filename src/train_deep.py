"""
Tier 2 — Deep learning models (CNN, LSTM, Transformer).

Shared training loop with:
  - Weighted cross-entropy + label smoothing (overfitting control)
  - CosineAnnealingLR scheduler
  - Early stopping with best-checkpoint restore
  - WeightedRandomSampler for class balance

Run standalone:
    python src/train_deep.py
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SEED, DEVICE,
    DL_BATCH_SIZE, DL_EPOCHS, DL_PATIENCE, DL_LR, DL_LR_MIN,
    DL_WEIGHT_DECAY, DL_LABEL_SMOOTH, DL_GRAD_CLIP,
    N_CLASSES, MODEL_DIR, OUTPUT_DIR,
)
from src.models import FaultTransformer, FaultCNN, FaultLSTM, FaultResNet1D
from src.utils import set_seed, get_logger, save_torch_model
from src.artifacts import build_artifact_registry
from src.preprocessing import load_preprocessed
from src.evaluate import evaluate_torch, compute_metrics, plot_confusion_matrix, plot_training_curves, plot_per_class_f1, save_metrics

log = get_logger("train_deep")


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    """Build DataLoaders. Sampler balances class frequency during training."""
    cw_np   = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    sw      = torch.tensor(cw_np[y_train], dtype=torch.float32)
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    tr_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long))
    va_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(y_val, dtype=torch.long))
    te_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                          torch.tensor(y_test, dtype=torch.long))

    kw = dict(num_workers=0, pin_memory=False)
    return (
        DataLoader(tr_ds, batch_size=batch_size, sampler=sampler, **kw),
        DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(te_ds, batch_size=batch_size, shuffle=False, **kw),
        cw_np,
    )


def train_one_model(model, train_loader, val_loader, cw_np,
                    model_name: str, epochs=DL_EPOCHS, patience=DL_PATIENCE):
    """
    Generic training loop for all deep learning models.
    Returns the best model and training history.
    """
    cw        = torch.tensor(cw_np, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=DL_LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DL_LR, weight_decay=DL_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=DL_LR_MIN
    )

    best_val_acc = 0.0
    best_state   = None
    wait         = 0
    tr_accs, va_accs     = [], []
    tr_losses, va_losses = [], []

    log.info(f"Training {model_name}  (max {epochs} epochs, patience {patience})")

    for epoch in range(1, epochs + 1):
        model.train()
        tp, tl, tloss = [], [], 0.0

        for xb, yb in tqdm(train_loader, desc=f"{model_name} E{epoch}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), DL_GRAD_CLIP)
            optimizer.step()
            tloss += loss.item()
            tp.extend(out.detach().argmax(1).cpu().numpy())
            tl.extend(yb.cpu().numpy())

        scheduler.step()

        from sklearn.metrics import accuracy_score
        tr_acc = accuracy_score(tl, tp)
        tr_accs.append(tr_acc)
        tr_losses.append(tloss / len(train_loader))

        model.eval()
        vp, vl, vloss = [], [], 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out  = model(xb)
                loss = criterion(out, yb)
                vloss += loss.item()
                vp.extend(out.argmax(1).cpu().numpy())
                vl.extend(yb.cpu().numpy())

        va_acc = accuracy_score(vl, vp)
        va_accs.append(va_acc)
        va_losses.append(vloss / len(val_loader))

        gap   = tr_acc - va_acc
        state = "OVERFIT" if gap > 0.08 else ("LOW" if va_acc < 0.65 else "OK")
        log.info(f"E{epoch:>3}  train={tr_acc:.4f} val={va_acc:.4f} gap={gap:+.4f} [{state}]")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = copy.deepcopy(model.state_dict())
            wait         = 0
        else:
            wait += 1
            if wait >= patience:
                log.info(f"Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    log.info(f"{model_name}  best val acc: {best_val_acc:.4f}")
    return model, tr_accs, va_accs, tr_losses, va_losses


def train_deep_models():
    set_seed(SEED)
    log.info("Loading preprocessed data")
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = load_preprocessed()
    n_feat = X_train.shape[1]

    train_loader, val_loader, test_loader, cw_np = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, DL_BATCH_SIZE
    )

    all_metrics = {}

    models_to_train = [
        ("Transformer", FaultTransformer(n_feat)),
        ("CNN",         FaultCNN(n_feat)),
        ("LSTM",        FaultLSTM(n_feat)),
        ("ResNet",      FaultResNet1D(n_feat)),
    ]

    for model_name, model in models_to_train:
        model = model.to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"{model_name} — {n_params:,} parameters")

        model, tr_accs, va_accs, tr_losses, va_losses = train_one_model(
            model, train_loader, val_loader, cw_np, model_name
        )

        save_torch_model(model, f"{model_name.lower()}_model.pth",
                         extra={"n_features": n_feat, "n_classes": N_CLASSES})

        metrics = evaluate_torch(model, X_test, y_test)
        from src.utils import print_metrics, save_metrics
        print_metrics(metrics, f"{model_name} — Test")
        save_metrics(metrics, f"{model_name.lower()}_metrics.json")

        plot_training_curves(
            tr_accs, va_accs, tr_losses, va_losses,
            model_name, f"{model_name.lower()}_curves.png"
        )
        plot_confusion_matrix(
            metrics["labels"], metrics["predictions"],
            f"{model_name} Confusion Matrix",
            f"{model_name.lower()}_confusion.png"
        )
        plot_per_class_f1(metrics, model_name, f"{model_name.lower()}_per_class_f1.png")
        all_metrics[model_name] = metrics

    build_artifact_registry()
    return all_metrics


if __name__ == "__main__":
    train_deep_models()
