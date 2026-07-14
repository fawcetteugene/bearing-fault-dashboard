"""
Artifact registry and production bundle helpers.

This module turns the project's loose set of saved model files into a single
versioned manifest. The dashboard and any inference entry point should read
from the production manifest instead of hardcoding filenames.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.config import (
    ALL_FEATURES,
    ARTIFACT_REGISTRY,
    CLASS_NAMES,
    ENCODER_PATH,
    MODEL_DIR,
    MODEL_METRICS_FILES,
    MODEL_ALIASES,
    N_CLASSES,
    PREPROCESSING_MANIFEST,
    PRODUCTION_MANIFEST,
    PRODUCTION_MODEL_PATH,
    PRODUCTION_TORCH_PATH,
    SCALER_PATH,
    SKLEARN_MODEL_FILES,
    TORCH_MODEL_FILES,
    TORCH_MODEL_ALIASES,
)


@dataclass
class ArtifactRecord:
    name: str
    kind: str
    model_path: str
    metrics_path: str | None
    metrics: dict[str, Any]
    file_size: int
    sha256: str
    updated_at_utc: str
    supported_for_dashboard: bool = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _guess_kind(name: str, model_path: str) -> str:
    if model_path.endswith(".pth"):
        return "torch"
    if name in SKLEARN_MODEL_FILES:
        return "sklearn"
    return "unknown"


def _resolve_model_path(name: str, preferred_filename: str) -> str | None:
    """Resolve a logical model name to the first existing filename alias."""
    aliases = MODEL_ALIASES.get(name, [preferred_filename])
    for filename in aliases:
        candidate = os.path.join(MODEL_DIR, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def _summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = {}
    for key in ("accuracy", "precision", "recall", "f1", "inference_ms",
                "best_val_acc", "average_accuracy", "task_results"):
        if key in metrics:
            keep[key] = metrics[key]
    return keep


def _normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state_dict, dict):
        return state_dict
    if state_dict and all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _repo_root() -> str:
    """Always returns the project root regardless of where the process is launched."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rel(path: str) -> str:
    """Store paths as repo-relative so they survive deployment to any mount point."""
    try:
        return os.path.relpath(path, _repo_root())
    except ValueError:
        return path  # different drive on Windows


def _abs(path: str) -> str:
    """Resolve a stored (possibly relative or stale-absolute) path to absolute at runtime."""
    if not os.path.isabs(path):
        return os.path.join(_repo_root(), path)
    # Stale absolute path from a different machine — remap via known repo-relative prefixes
    for prefix in ("models/", "outputs/", "data/"):
        idx = path.find("/" + prefix)
        if idx != -1:
            candidate = os.path.join(_repo_root(), path[idx + 1:])
            if os.path.exists(candidate):
                return candidate
    return path  # last resort: use as-is


def _candidate_from_paths(name: str, model_path: str, metrics_path: str | None) -> ArtifactRecord | None:
    if not os.path.exists(model_path):
        return None
    metrics = _summarize_metrics(_read_json(metrics_path) if metrics_path else {})
    supported = name in {"rf", "xgb", "transformer", "maml", "meta_sgd", "fbcl"}
    return ArtifactRecord(
        name=name,
        kind=_guess_kind(name, model_path),
        model_path=_rel(model_path),
        metrics_path=_rel(metrics_path) if metrics_path and os.path.exists(metrics_path) else None,
        metrics=metrics,
        file_size=os.path.getsize(model_path),
        sha256=_sha256(model_path),
        updated_at_utc=_utc_now(),
        supported_for_dashboard=supported,
    )


def discover_artifacts() -> list[ArtifactRecord]:
    candidates: list[ArtifactRecord] = []
    merged = {**SKLEARN_MODEL_FILES, **TORCH_MODEL_FILES}
    for name, filename in merged.items():
        model_path = _resolve_model_path(name, filename)
        if model_path is None:
            continue
        metrics_path = os.path.join(os.path.dirname(MODEL_DIR), "outputs", MODEL_METRICS_FILES.get(name, ""))
        record = _candidate_from_paths(name, model_path, metrics_path)
        if record is not None:
            candidates.append(record)
    return candidates


def _score_candidate(candidate: ArtifactRecord) -> tuple:
    metrics = candidate.metrics or {}
    primary = metrics.get("accuracy", metrics.get("best_val_acc", 0.0))
    secondary = metrics.get("f1", metrics.get("average_accuracy", 0.0))
    latency = metrics.get("inference_ms", float("inf"))
    return (float(primary), float(secondary), -float(latency))


def select_production_candidate(candidates: list[ArtifactRecord] | None = None) -> ArtifactRecord | None:
    candidates = candidates or discover_artifacts()
    selectable = [c for c in candidates if c.supported_for_dashboard]
    if not selectable:
        return None
    return sorted(selectable, key=_score_candidate, reverse=True)[0]


def _copy_production_model(candidate: ArtifactRecord) -> str:
    destination = PRODUCTION_MODEL_PATH if candidate.kind == "sklearn" else PRODUCTION_TORCH_PATH
    shutil.copy2(candidate.model_path, destination)
    return destination


def build_artifact_registry(force: bool = False) -> dict[str, Any]:
    """
    Build and persist a single registry describing all saved model artifacts.
    Also promotes the best supported model into the production bundle.
    """
    candidates = discover_artifacts()
    production = select_production_candidate(candidates)
    registry = {
        "created_at_utc": _utc_now(),
        "seed": 42,
        "feature_order": ALL_FEATURES,
        "target_classes": CLASS_NAMES,
        "scaler_path": _rel(SCALER_PATH) if os.path.exists(SCALER_PATH) else None,
        "encoder_path": _rel(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None,
        "preprocessing_manifest_path": _rel(PREPROCESSING_MANIFEST) if os.path.exists(PREPROCESSING_MANIFEST) else None,
        "candidates": [asdict(c) for c in candidates],
        "production": asdict(production) if production else None,
    }

    if production:
        registry["production"]["production_bundle_path"] = _rel(_copy_production_model(production))
    if force or not os.path.exists(ARTIFACT_REGISTRY):
        with open(ARTIFACT_REGISTRY, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    else:
        with open(ARTIFACT_REGISTRY, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    with open(PRODUCTION_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    return registry


def load_artifact_registry() -> dict[str, Any]:
    if os.path.exists(PRODUCTION_MANIFEST):
        return _read_json(PRODUCTION_MANIFEST)
    if os.path.exists(ARTIFACT_REGISTRY):
        return _read_json(ARTIFACT_REGISTRY)
    return build_artifact_registry(force=True)


def load_preprocessing_manifest() -> dict[str, Any]:
    return _read_json(PREPROCESSING_MANIFEST)


def _load_torch_model(record: dict[str, Any]):
    import torch
    from src.models import (
        FaultCNN, FaultLSTM, FaultTransformer, FaultResNet1D,
        MAMLCheckpointModel, MetaSGDCheckpointModel, FBCLCheckpointModel,
    )

    try:
        payload = torch.load(record["model_path"], map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Could not load checkpoint {record['model_path']}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint format in {record['model_path']}")

    state_dict = (
        payload.get("model_state_dict")
        or payload.get("state_dict")
        or payload.get("model")
        or payload.get("net")
    )
    if state_dict is None and all(hasattr(v, "shape") for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"No state_dict found in {record['model_path']}")
    state_dict = _normalize_state_dict(state_dict)

    # Use n_features stored in the checkpoint — checkpoints were trained on 19 features.
    n_features = payload.get("n_features", len(ALL_FEATURES))
    n_classes  = payload.get("n_classes", N_CLASSES)

    name = record["name"]
    if name == "transformer":
        model = FaultTransformer(n_features, n_classes=n_classes)
    elif name == "cnn":
        model = FaultCNN(n_features, n_classes=n_classes)
    elif name == "lstm":
        model = FaultLSTM(n_features, n_classes=n_classes)
    elif name == "resnet":
        model = FaultResNet1D(n_features, n_classes=n_classes)
    elif name == "maml":
        model = MAMLCheckpointModel(n_features, n_classes)
    elif name == "meta_sgd":
        model = MetaSGDCheckpointModel(n_features, n_classes)
    elif name == "fbcl":
        model = FBCLCheckpointModel(n_features, n_classes)
    else:
        raise ValueError(f"Unsupported torch artifact for dashboard: {name}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_artifact_model(model_name: str | None = None):
    """
    Load the production model or a named archived model from the registry.
    """
    registry = load_artifact_registry()
    candidates = registry.get("candidates", [])

    record = None
    if model_name:
        for item in candidates:
            if item["name"] == model_name:
                record = item
                break
        if record is None and model_name in MODEL_ALIASES:
            model_path = _resolve_model_path(model_name, MODEL_ALIASES[model_name][0])
            if model_path:
                record = {
                    "name": model_name,
                    "kind": _guess_kind(model_name, model_path),
                    "model_path": model_path,
                }
    else:
        record = registry.get("production")

    if not record:
        raise FileNotFoundError("No matching model artifact found.")

    if record["kind"] == "sklearn":
        return joblib.load(_abs(record["model_path"]))
    if record["kind"] == "torch":
        record = dict(record)
        record["model_path"] = _abs(record["model_path"])
        return _load_torch_model(record)
    raise ValueError(f"Unsupported artifact kind: {record['kind']}")


def compare_artifacts(candidates: list[ArtifactRecord] | None = None) -> list[dict[str, Any]]:
    candidates = candidates or discover_artifacts()
    rows = []
    for cand in candidates:
        metrics = cand.metrics or {}
        rows.append({
            "name": cand.name,
            "kind": cand.kind,
            "accuracy": metrics.get("accuracy", metrics.get("best_val_acc", None)),
            "f1": metrics.get("f1", metrics.get("average_accuracy", None)),
            "precision": metrics.get("precision", None),
            "recall": metrics.get("recall", None),
            "inference_ms": metrics.get("inference_ms", None),
            "supported_for_dashboard": cand.supported_for_dashboard,
            "path": cand.model_path,
            "sha256": cand.sha256,
        })
    return sorted(rows, key=lambda r: (r["accuracy"] or 0.0, r["f1"] or 0.0), reverse=True)
