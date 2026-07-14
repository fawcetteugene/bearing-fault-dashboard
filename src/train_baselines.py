"""
Tier 1 — Baseline ensemble models.

Trains Random Forest, XGBoost, GradientBoosting, LightGBM, and a stacking
ensemble on the augmented training set with RandomizedSearchCV tuning.

Run standalone:
    python src/train_baselines.py
"""

import os
import sys
import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SEED, RF_PARAMS, XGB_PARAMS, GBM_PARAMS, LGBM_PARAMS, MODEL_DIR
from src.utils import set_seed, get_logger, save_sklearn_model, save_metrics, print_metrics
from src.artifacts import build_artifact_registry
from src.preprocessing import load_preprocessed
from src.evaluate import compute_metrics

log = get_logger("train_baselines")


# Search spaces for RandomizedSearchCV
RF_SEARCH_SPACE = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
}

XGB_SEARCH_SPACE = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 5, 6, 8],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

GBM_SEARCH_SPACE = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [3, 4, 5, 6],
    "learning_rate":   [0.01, 0.05, 0.1],
    "subsample":       [0.7, 0.8, 1.0],
    "min_samples_split": [2, 5, 10],
}

LGBM_SEARCH_SPACE = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [4, 6, 8, -1],
    "learning_rate":   [0.01, 0.05, 0.1],
    "num_leaves":      [31, 63, 127],
    "subsample":       [0.7, 0.8, 1.0],
    "colsample_bytree":[0.7, 0.8, 1.0],
}


def tune_and_train(base_model, search_space: dict, X_train, y_train,
                   model_name: str, n_iter: int = 20):
    """
    Run RandomizedSearchCV then return the best estimator.
    Uses 3-fold cross-validation for speed on the large training set.
    """
    log.info(f"Tuning {model_name} ({n_iter} iterations, 3-fold CV)")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    # Use a random 20% subsample of training data for speed during tuning
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_train), size=min(20_000, len(X_train)), replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]

    search = RandomizedSearchCV(
        base_model, search_space,
        n_iter=n_iter, cv=cv, scoring="accuracy",
        n_jobs=-1, random_state=SEED, verbose=0,
    )
    search.fit(X_sub, y_sub)
    log.info(f"{model_name} best CV accuracy: {search.best_score_:.4f}")
    log.info(f"{model_name} best params: {search.best_params_}")

    # Retrain best config on full training set
    best = search.best_estimator_
    best.fit(X_train, y_train)
    return best


def train_baselines():
    set_seed(SEED)
    log.info("Loading preprocessed data")
    X_train, y_train, X_val, y_val, X_test, y_test, _, encoder = load_preprocessed()

    results = {}

    # Random Forest
    rf = tune_and_train(
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        RF_SEARCH_SPACE, X_train, y_train, "RandomForest",
    )
    save_sklearn_model(rf, "rf_model.pkl")
    t0 = time.perf_counter()
    rf_metrics = compute_metrics(rf.predict(X_test), y_test)
    rf_metrics["inference_ms"] = (time.perf_counter() - t0) / len(X_test) * 1000
    print_metrics(rf_metrics, "Random Forest — Test")
    save_metrics(rf_metrics, "rf_metrics.json")
    results["random_forest"] = rf_metrics

    # XGBoost
    xgb = tune_and_train(
        XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                      random_state=SEED, n_jobs=-1),
        XGB_SEARCH_SPACE, X_train, y_train, "XGBoost",
    )
    save_sklearn_model(xgb, "xgb_model.pkl")
    t0 = time.perf_counter()
    xgb_metrics = compute_metrics(xgb.predict(X_test), y_test)
    xgb_metrics["inference_ms"] = (time.perf_counter() - t0) / len(X_test) * 1000
    print_metrics(xgb_metrics, "XGBoost — Test")
    save_metrics(xgb_metrics, "xgb_metrics.json")
    results["xgboost"] = xgb_metrics

    # Gradient Boosting
    gbm = tune_and_train(
        GradientBoostingClassifier(random_state=SEED),
        GBM_SEARCH_SPACE, X_train, y_train, "GBM",
    )
    save_sklearn_model(gbm, "gbm_model.pkl")
    t0 = time.perf_counter()
    gbm_metrics = compute_metrics(gbm.predict(X_test), y_test)
    gbm_metrics["inference_ms"] = (time.perf_counter() - t0) / len(X_test) * 1000
    print_metrics(gbm_metrics, "GBM — Test")
    save_metrics(gbm_metrics, "gbm_metrics.json")
    results["gbm"] = gbm_metrics

    # LightGBM
    lgbm = tune_and_train(
        LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1),
        LGBM_SEARCH_SPACE, X_train, y_train, "LightGBM",
    )
    save_sklearn_model(lgbm, "lgbm_model.pkl")
    t0 = time.perf_counter()
    lgbm_metrics = compute_metrics(lgbm.predict(X_test), y_test)
    lgbm_metrics["inference_ms"] = (time.perf_counter() - t0) / len(X_test) * 1000
    print_metrics(lgbm_metrics, "LightGBM — Test")
    save_metrics(lgbm_metrics, "lgbm_metrics.json")
    results["lgbm"] = lgbm_metrics

    # Stacking ensemble (RF + XGB + LGBM → Logistic Regression meta-learner)
    log.info("Training stacking ensemble")
    stack = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=3, n_jobs=-1, passthrough=False,
    )
    # Fit on a subsample for speed (stacking CV is expensive on 100k rows)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_train), size=min(30_000, len(X_train)), replace=False)
    stack.fit(X_train[idx], y_train[idx])
    save_sklearn_model(stack, "stack_model.pkl")
    t0 = time.perf_counter()
    stack_metrics = compute_metrics(stack.predict(X_test), y_test)
    stack_metrics["inference_ms"] = (time.perf_counter() - t0) / len(X_test) * 1000
    print_metrics(stack_metrics, "Stacking Ensemble — Test")
    save_metrics(stack_metrics, "stack_metrics.json")
    results["stack"] = stack_metrics

    # Summary table
    print("\nBaseline Summary")
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>8} {'ms/sample':>12}")
    for name, m in results.items():
        print(f"{name:<20} {m['accuracy']:>10.4f} {m['f1']:>8.4f} {m['inference_ms']:>12.4f}")

    build_artifact_registry()
    return results


if __name__ == "__main__":
    train_baselines()
