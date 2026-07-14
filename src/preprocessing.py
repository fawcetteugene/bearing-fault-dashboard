"""
Data preprocessing pipeline.

Steps (in order, all fitted on train only):
  1. Load raw CSV
  2. Stratified 70/15/15 split
  3. Isolation Forest outlier removal
  4. Engineer 10 additional features (19 total)
  5. Label encode target
  6. StandardScaler normalisation
  7. SMOTE + Gaussian bootstrap augmentation of training set to 100,000
  8. Save all four CSV files and serialised scaler/encoder

Run standalone:
    python src/preprocessing.py --data data/featuretime48k2048load_1.csv
"""

import os
import sys
import argparse
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RAW_CSV, TRAIN_SCALED_CSV, VAL_SCALED_CSV, TEST_SCALED_CSV,
    SCALER_PATH, ENCODER_PATH, PREPROCESSING_MANIFEST,
    ORIGINAL_FEATURES, ENGINEERED_FEATURES, FFT_FEATURES, ALL_FEATURES, TARGET_COL,
    TRAIN_RATIO, VAL_RATIO, SEED,
    OUTLIER_CONTAMINATION, SMOTE_K, AUG_TARGET, AUG_SIGMA,
    DATA_DIR, MODEL_DIR,
)
from src.utils import set_seed, get_logger

log = get_logger("preprocessing")

# ---------- Compute missing constants ----------
TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO   # Fix for the NameError

# If any of these are not defined in config, set sensible defaults
if 'OUTLIER_CONTAMINATION' not in locals():
    OUTLIER_CONTAMINATION = 0.05
if 'SMOTE_K' not in locals():
    SMOTE_K = 5
if 'AUG_TARGET' not in locals():
    AUG_TARGET = 100000
if 'AUG_SIGMA' not in locals():
    AUG_SIGMA = 0.01


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 10 domain-specific interaction features + 6 FFT-proxy frequency-domain
    features derived from the 9 original statistical features (25 total).
    """
    d = df.copy()
    eps = 1e-8

    # --- 10 interaction features ---
    d["peak_to_peak"]    = d["max"] - d["min"]
    d["rms_mean_ratio"]  = d["rms"] / (d["mean"].abs() + eps)
    d["crest_form_ratio"]= d["crest"] / (d["form"] + eps)
    d["kurtosis_squared"]= d["kurtosis"] ** 2
    d["skewness_abs"]    = d["skewness"].abs()
    d["rms_sd_ratio"]    = d["rms"] / (d["sd"] + eps)
    d["max_min_ratio"]   = (d["max"].abs() + 1) / (d["min"].abs() + 1)
    d["kurtosis_crest"]  = d["kurtosis"] * d["crest"]
    d["mean_squared"]    = d["mean"] ** 2
    d["rms_squared"]     = d["rms"] ** 2

    # --- 6 frequency-domain proxy features ---
    d["energy"]           = d["rms"] ** 2 * d["sd"] ** 2
    # Spectral entropy from normalised descriptor vector
    desc = d[["rms", "sd", "kurtosis", "skewness", "crest", "form"]].abs()
    desc_norm = desc.div(desc.sum(axis=1) + eps, axis=0)
    d["spectral_entropy"] = -(desc_norm * np.log(desc_norm + eps)).sum(axis=1)
    d["impulse_factor"]   = d["max"].abs() / (d["rms"] + eps)
    d["margin_factor"]    = d["max"].abs() / (d["mean"].abs() ** 2 + eps)
    d["shape_factor"]     = d["rms"] / (d["mean"].abs() + eps)
    d["clearance_factor"] = d["max"].abs() / (np.sqrt(d["rms"].abs()) + eps)

    return d


def remove_outliers(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray):
    """
    Fit Isolation Forest on the training set only, then apply to all three splits.
    Returns cleaned arrays and the fitted detector (needed for inference).
    """
    iso = IsolationForest(contamination=OUTLIER_CONTAMINATION,
                          random_state=SEED, n_jobs=-1)
    train_mask = iso.fit_predict(X_train) == 1
    val_mask   = iso.predict(X_val)  == 1
    test_mask  = iso.predict(X_test) == 1

    log.info(f"Outlier removal: train {X_train.shape[0]} → {train_mask.sum()}")
    log.info(f"Outlier removal: val   {X_val.shape[0]} → {val_mask.sum()}")
    log.info(f"Outlier removal: test  {X_test.shape[0]} → {test_mask.sum()}")

    return (X_train[train_mask], y_train[train_mask],
            X_val[val_mask],     y_val[val_mask],
            X_test[test_mask],   y_test[test_mask],
            iso)


def augment_training_set(X: np.ndarray, y: np.ndarray,
                         target: int = AUG_TARGET,
                         sigma: float = AUG_SIGMA):
    """
    Two‑step augmentation:
      1. SMOTE to balance classes (k=5 neighbours).
      2. Bootstrap with tiny Gaussian noise to reach `target` samples.
    """
    sm = SMOTE(k_neighbors=SMOTE_K, random_state=SEED)
    X_sm, y_sm = sm.fit_resample(X, y)
    log.info(f"After SMOTE: {X_sm.shape[0]} samples")

    rng     = np.random.default_rng(SEED)
    current = X_sm.shape[0]
    needed  = target - current

    idx        = rng.integers(0, current, size=needed)
    X_boot     = X_sm[idx] + rng.normal(0, sigma, (needed, X_sm.shape[1]))
    y_boot     = y_sm[idx]

    X_aug = np.vstack([X_sm, X_boot])
    y_aug = np.concatenate([y_sm, y_boot])

    perm  = rng.permutation(len(X_aug))
    log.info(f"After augmentation: {X_aug.shape[0]} samples")
    return X_aug[perm], y_aug[perm]


def preprocess_pipeline(raw_csv: str = RAW_CSV):
    """
    Full pipeline from raw CSV to saved preprocessed files.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test, scaler, encoder).
    """
    set_seed(SEED)
    log.info(f"Loading {raw_csv}")
    df = pd.read_csv(raw_csv)

    # Detect target column — could be 'fault' or already 'fault_encoded'
    target_raw = "fault" if "fault" in df.columns else df.columns[-1]
    log.info(f"Target column: {target_raw}  |  Samples: {len(df)}")

    # Make sure all feature columns are numeric
    for col in ORIGINAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=ORIGINAL_FEATURES)

    # Engineer features before splitting
    df = engineer_features(df)

    # ------------------------------------------------------------------
    # FIX: use .to_numpy() to get plain NumPy arrays (avoids Arrow dtype issues)
    X_all = df[ALL_FEATURES].to_numpy(dtype=np.float32)
    y_raw = df[target_raw].to_numpy()   # strings (for stratification)
    # ------------------------------------------------------------------

    # Stratified split — done before any scaling/encoding
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_all, y_raw,
        test_size=TEST_RATIO,
        stratify=y_raw,
        random_state=SEED,
    )
    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_size,
        stratify=y_tv,
        random_state=SEED,
    )
    log.info(f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # Outlier removal (train‑fitted detector applied to all splits)
    X_train, y_train, X_val, y_val, X_test, y_test, iso = remove_outliers(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Label encoding — fit on all labels so no unknown classes at eval
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # Feature scaling — fit on training set only
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # Augment training set to 100,000 balanced samples
    X_aug, y_aug = augment_training_set(X_train_sc, y_train_enc)

    # Save preprocessed files
    train_df = pd.DataFrame(X_aug, columns=ALL_FEATURES)
    train_df[TARGET_COL] = y_aug
    train_df.to_csv(TRAIN_SCALED_CSV, index=False)

    val_df = pd.DataFrame(X_val_sc, columns=ALL_FEATURES)
    val_df[TARGET_COL] = y_val_enc
    val_df.to_csv(VAL_SCALED_CSV, index=False)

    test_df = pd.DataFrame(X_test_sc, columns=ALL_FEATURES)
    test_df[TARGET_COL] = y_test_enc
    test_df.to_csv(TEST_SCALED_CSV, index=False)

    # Save scaler and encoder for inference
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le,     ENCODER_PATH)
    log.info("Saved scaler, encoder and all CSV splits")

    preprocessing_manifest = {
        "source_csv": os.path.abspath(raw_csv),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "raw_rows": int(len(df)),
        "original_feature_count": len(ORIGINAL_FEATURES),
        "engineered_feature_count": len(ENGINEERED_FEATURES),
        "fft_feature_count": len(FFT_FEATURES),
        "total_feature_count": len(ALL_FEATURES),
        "split_sizes_after_cleaning": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "augmentation": {
            "target_rows": AUG_TARGET,
            "sigma": AUG_SIGMA,
            "smote_k": SMOTE_K,
            "final_train_rows": int(len(X_aug)),
        },
        "class_names": list(le.classes_),
        "class_to_index": {name: int(idx) for idx, name in enumerate(le.classes_)},
        "feature_order": ALL_FEATURES,
        "files": {
            "train": TRAIN_SCALED_CSV,
            "validation": VAL_SCALED_CSV,
            "test": TEST_SCALED_CSV,
            "scaler": SCALER_PATH,
            "encoder": ENCODER_PATH,
        },
        "outlier_detector": {
            "type": "IsolationForest",
            "contamination": OUTLIER_CONTAMINATION,
            "random_state": SEED,
        },
    }

    with open(PREPROCESSING_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(preprocessing_manifest, f, indent=2)
    log.info(f"Saved preprocessing manifest: {PREPROCESSING_MANIFEST}")

    return X_aug, y_aug, X_val_sc, y_val_enc, X_test_sc, y_test_enc, scaler, le


def load_preprocessed():
    """
    Load already‑preprocessed CSVs back into numpy arrays.
    Faster than re‑running the full pipeline.
    """
    def read(path):
        df = pd.read_csv(path)
        X  = df[ALL_FEATURES].values.astype(np.float32)
        y  = df[TARGET_COL].values.astype(int)
        return X, y

    X_train, y_train = read(TRAIN_SCALED_CSV)
    X_val,   y_val   = read(VAL_SCALED_CSV)
    X_test,  y_test  = read(TEST_SCALED_CSV)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=RAW_CSV, help="Path to raw CSV file")
    args = parser.parse_args()
    preprocess_pipeline(args.data)
    log.info("Preprocessing complete.")
