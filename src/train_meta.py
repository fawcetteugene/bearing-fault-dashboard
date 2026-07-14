"""
Tier 3a — Meta-learning: MAML and Meta-SGD.

Episode-based training where each episode:
  1. Samples N classes (N-way), K support examples per class
  2. Inner loop: adapts a copy of the model on the support set
  3. Outer loop: meta-update using query set loss

Meta-SGD adds learnable per-parameter inner-loop learning rates on top of MAML.

Run standalone:
    python src/train_meta.py
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SEED, DEVICE, N_CLASSES,
    META_N_WAY, META_K_SHOT, META_Q_QUERY,
    META_INNER_LR, META_OUTER_LR, META_INNER_STEPS,
    META_EPISODES, META_EVAL_EPISODES,
)
from src.models import MetaLearnerBase
from src.utils import set_seed, get_logger, save_torch_model, save_metrics
from src.artifacts import build_artifact_registry
from src.preprocessing import load_preprocessed

log = get_logger("train_meta")


def sample_episode(X: np.ndarray, y: np.ndarray,
                   n_way: int, k_shot: int, q_query: int):
    """
    Sample one N-way K-shot episode.
    Returns support and query tensors for a randomly chosen subset of classes.
    """
    classes = np.random.choice(np.unique(y), size=n_way, replace=False)
    # Remap chosen classes to 0..n_way-1 for the inner loop loss
    label_map = {c: i for i, c in enumerate(classes)}

    sx, sy, qx, qy = [], [], [], []
    for c in classes:
        idx     = np.where(y == c)[0]
        chosen  = np.random.choice(idx, size=k_shot + q_query, replace=False)
        support = chosen[:k_shot]
        query   = chosen[k_shot:]
        sx.append(X[support]);  sy.extend([label_map[c]] * k_shot)
        qx.append(X[query]);    qy.extend([label_map[c]] * q_query)

    sx = torch.tensor(np.vstack(sx), dtype=torch.float32).to(DEVICE)
    sy = torch.tensor(sy, dtype=torch.long).to(DEVICE)
    qx = torch.tensor(np.vstack(qx), dtype=torch.float32).to(DEVICE)
    qy = torch.tensor(qy, dtype=torch.long).to(DEVICE)
    return sx, sy, qx, qy, n_way


def inner_loop_maml(model: nn.Module, sx, sy, n_way: int,
                    inner_lr: float, inner_steps: int):
    """
    MAML inner loop: clone the model, run `inner_steps` gradient descent
    steps on the support set, and return the adapted clone.
    """
    fast_model = copy.deepcopy(model)
    fast_opt   = torch.optim.SGD(fast_model.parameters(), lr=inner_lr)

    for _ in range(inner_steps):
        fast_opt.zero_grad()
        logits = fast_model(sx)
        # Only use the n_way output units (the model has N_CLASSES outputs;
        # slice the first n_way for this episode)
        loss = F.cross_entropy(logits[:, :n_way], sy)
        loss.backward()
        fast_opt.step()

    return fast_model


class MetaSGD(nn.Module):
    """
    Meta-SGD wraps the base learner and adds a learnable log learning rate
    per parameter. These are meta-learned alongside the initial weights.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        # Initialise log-LR parameters (learned per-param step sizes)
        self.log_lrs = nn.ParameterList([
            nn.Parameter(torch.full_like(p.data, fill_value=-2.0))
            for p in base_model.parameters()
        ])

    def get_lrs(self):
        return [torch.exp(log_lr) for log_lr in self.log_lrs]

    def adapted_params(self, sx, sy, n_way: int):
        """One inner-loop step with learned per-param LRs."""
        logits = self.base(sx)
        loss   = F.cross_entropy(logits[:, :n_way], sy)
        grads  = torch.autograd.grad(loss, self.base.parameters(),
                                     create_graph=True, allow_unused=True)
        adapted = []
        lrs     = self.get_lrs()
        for p, g, lr in zip(self.base.parameters(), grads, lrs):
            adapted.append(p - lr * g if g is not None else p)
        return adapted

    def forward_with_params(self, x, params):
        """Run forward pass using adapted parameter list (functional API)."""
        # Simple implementation: temporarily swap params
        original = [p.data.clone() for p in self.base.parameters()]
        for p, adapted in zip(self.base.parameters(), params):
            p.data = adapted.data
        out = self.base(x)
        for p, orig in zip(self.base.parameters(), original):
            p.data = orig
        return out


def train_maml(X_train, y_train, X_val, y_val, n_features: int):
    set_seed(SEED)
    base  = MetaLearnerBase(n_features, META_N_WAY).to(DEVICE)
    meta_opt = torch.optim.Adam(base.parameters(), lr=META_OUTER_LR)

    log.info(f"MAML training — {META_EPISODES} episodes, {META_N_WAY}-way {META_K_SHOT}-shot")
    best_val_acc = 0.0
    best_state   = None
    episode_losses = []

    for ep in tqdm(range(1, META_EPISODES + 1), desc="MAML episodes"):
        sx, sy, qx, qy, n_way = sample_episode(
            X_train, y_train, META_N_WAY, META_K_SHOT, META_Q_QUERY
        )
        # Inner loop adaptation
        fast = inner_loop_maml(base, sx, sy, n_way, META_INNER_LR, META_INNER_STEPS)

        # Outer loop: query loss through the adapted model
        meta_opt.zero_grad()
        q_logits = fast(qx)[:, :n_way]
        meta_loss = F.cross_entropy(q_logits, qy)
        meta_loss.backward()
        meta_opt.step()
        episode_losses.append(meta_loss.item())

        # Evaluate on validation set every 50 episodes
        if ep % 50 == 0:
            val_accs = []
            for _ in range(50):
                vsx, vsy, vqx, vqy, vn = sample_episode(
                    X_val, y_val, META_N_WAY, META_K_SHOT, META_Q_QUERY
                )
                fast_v = inner_loop_maml(base, vsx, vsy, vn,
                                         META_INNER_LR, META_INNER_STEPS)
                with torch.no_grad():
                    preds  = fast_v(vqx)[:, :vn].argmax(1)
                val_accs.append((preds == vqy).float().mean().item())
            val_acc = float(np.mean(val_accs))
            log.info(f"Episode {ep}  meta_loss={np.mean(episode_losses[-50:]):.4f}  "
                     f"val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = copy.deepcopy(base.state_dict())

    base.load_state_dict(best_state)
    save_torch_model(base, "maml_model.pth",
                     extra={"n_features": n_features, "n_way": META_N_WAY})
    metrics = {"best_val_acc": best_val_acc}
    save_metrics(metrics, "maml_metrics.json")
    log.info(f"MAML best val acc: {best_val_acc:.4f}")
    return base, metrics


def train_meta_sgd(X_train, y_train, X_val, y_val, n_features: int):
    set_seed(SEED)
    base     = MetaLearnerBase(n_features, META_N_WAY).to(DEVICE)
    meta_sgd = MetaSGD(base).to(DEVICE)
    meta_opt = torch.optim.Adam(meta_sgd.parameters(), lr=META_OUTER_LR)

    log.info(f"Meta-SGD training — {META_EPISODES} episodes")
    best_val_acc = 0.0
    best_state   = None

    for ep in tqdm(range(1, META_EPISODES + 1), desc="Meta-SGD episodes"):
        sx, sy, qx, qy, n_way = sample_episode(
            X_train, y_train, META_N_WAY, META_K_SHOT, META_Q_QUERY
        )
        # Single inner-loop step with learned LRs
        adapted_params = meta_sgd.adapted_params(sx, sy, n_way)

        meta_opt.zero_grad()
        q_logits = meta_sgd.forward_with_params(qx, adapted_params)[:, :n_way]
        meta_loss = F.cross_entropy(q_logits, qy)
        meta_loss.backward()
        meta_opt.step()

        if ep % 50 == 0:
            val_accs = []
            for _ in range(50):
                vsx, vsy, vqx, vqy, vn = sample_episode(
                    X_val, y_val, META_N_WAY, META_K_SHOT, META_Q_QUERY
                )
                adapted = meta_sgd.adapted_params(vsx, vsy, vn)
                with torch.no_grad():
                    preds = meta_sgd.forward_with_params(vqx, adapted)[:, :vn].argmax(1)
                val_accs.append((preds == vqy).float().mean().item())
            val_acc = float(np.mean(val_accs))
            log.info(f"Episode {ep}  val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = copy.deepcopy(meta_sgd.state_dict())

    meta_sgd.load_state_dict(best_state)
    save_torch_model(meta_sgd, "meta_sgd_model.pth",
                     extra={"n_features": n_features, "n_way": META_N_WAY})
    metrics = {"best_val_acc": best_val_acc}
    save_metrics(metrics, "meta_sgd_metrics.json")
    log.info(f"Meta-SGD best val acc: {best_val_acc:.4f}")
    return meta_sgd, metrics


def train_meta_models():
    set_seed(SEED)
    X_train, y_train, X_val, y_val, _, _, _, _ = load_preprocessed()
    n_feat = X_train.shape[1]

    all_metrics = {}
    _, maml_m     = train_maml(X_train, y_train, X_val, y_val, n_feat)
    _, meta_sgd_m = train_meta_sgd(X_train, y_train, X_val, y_val, n_feat)

    all_metrics["MAML"]     = maml_m
    all_metrics["Meta-SGD"] = meta_sgd_m
    build_artifact_registry()
    return all_metrics


if __name__ == "__main__":
    train_meta_models()
