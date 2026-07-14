"""
Central configuration file.
Every hyperparameter, path and constant lives here.
Import this module everywhere else — nothing is hardcoded.
"""

import os

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# Raw dataset filename (place in data/)
RAW_CSV = os.path.join(DATA_DIR, "featuretime48k2048load_1.csv")

# Preprocessed file names written by preprocessing.py
TRAIN_SCALED_CSV = os.path.join(DATA_DIR, "preprocessed_augmented_dataset_scaled.csv")
VAL_SCALED_CSV   = os.path.join(DATA_DIR, "validation_set_scaled.csv")
TEST_SCALED_CSV  = os.path.join(DATA_DIR, "test_set_scaled.csv")
PREPROCESSING_MANIFEST = os.path.join(MODEL_DIR, "preprocessing_manifest.json")
ARTIFACT_REGISTRY      = os.path.join(MODEL_DIR, "artifact_registry.json")
PRODUCTION_MANIFEST    = os.path.join(MODEL_DIR, "production_manifest.json")
PRODUCTION_MODEL_PATH  = os.path.join(MODEL_DIR, "production_model.pkl")
PRODUCTION_TORCH_PATH  = os.path.join(MODEL_DIR, "production_model.pth")
SCALER_PATH      = os.path.join(MODEL_DIR, "feature_scaler.pkl")
ENCODER_PATH     = os.path.join(MODEL_DIR, "label_encoder.pkl")

TARGET_COL = "fault_encoded"

# Original 9 features from the CWRU CSV
ORIGINAL_FEATURES = ["max", "min", "mean", "sd", "rms",
                     "skewness", "kurtosis", "crest", "form"]

# 10 engineered features added during preprocessing
ENGINEERED_FEATURES = [
    "peak_to_peak", "rms_mean_ratio", "crest_form_ratio",
    "kurtosis_squared", "skewness_abs", "rms_sd_ratio",
    "max_min_ratio", "kurtosis_crest", "mean_squared", "rms_squared",
]

# 6 FFT-proxy frequency-domain features derived from statistical features
FFT_FEATURES = [
    "energy",          # rms^2 * sd^2  — proxy for signal energy
    "spectral_entropy",# entropy of normalised [rms, sd, kurtosis, skewness, crest, form]
    "impulse_factor",  # max / rms
    "margin_factor",   # max / mean^2
    "shape_factor",    # rms / mean_abs
    "clearance_factor",# max / rms_squared_sqrt
]

ALL_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES + FFT_FEATURES  # 25 total

# Fault class names (sorted alphabetically = label encoder output order)
CLASS_NAMES = [
    "Ball_007_1", "Ball_014_1", "Ball_021_1",
    "IR_007_1",   "IR_014_1",   "IR_021_1",
    "Normal_1",
    "OR_007_6_1", "OR_014_6_1", "OR_021_6_1",
]
N_CLASSES = len(CLASS_NAMES)   # 10

# Reproducibility
SEED = 42

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Isolation Forest contamination for outlier removal
OUTLIER_CONTAMINATION = 0.05

# SMOTE k-neighbours
SMOTE_K = 5

# Bootstrapping target total samples and Gaussian noise sigma
AUG_TARGET  = 100_000
AUG_SIGMA   = 0.005

# Device — auto-detect GPU
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"


# Baseline model hyperparameters (tuned via RandomizedSearchCV)
RF_PARAMS = {
    "n_estimators":      300,
    "max_depth":         None,
    "min_samples_split": 2,
    "min_samples_leaf":  1,
    "max_features":      "sqrt",
    "class_weight":      "balanced",
    "random_state":      SEED,
    "n_jobs":            -1,
}

XGB_PARAMS = {
    "n_estimators":    300,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "use_label_encoder": False,
    "eval_metric":     "mlogloss",
    "random_state":    SEED,
    "n_jobs":          -1,
}

GBM_PARAMS = {
    "n_estimators":    200,
    "max_depth":       5,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "min_samples_split": 4,
    "random_state":    SEED,
}

LGBM_PARAMS = {
    "n_estimators":    300,
    "max_depth":       6,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "class_weight":    "balanced",
    "random_state":    SEED,
    "n_jobs":          -1,
    "verbose":         -1,
}


# Deep learning shared hyperparameters
DL_BATCH_SIZE   = 128
DL_EPOCHS       = 100
DL_PATIENCE     = 15
DL_LR           = 3e-4
DL_LR_MIN       = 1e-6
DL_WEIGHT_DECAY = 3e-4
DL_LABEL_SMOOTH = 0.07
DL_GRAD_CLIP    = 2.0

# Transformer architecture
TRANS_D_MODEL  = 64
TRANS_N_HEADS  = 4
TRANS_N_LAYERS = 3
TRANS_DROPOUT  = 0.12

# 1D CNN architecture
CNN_CHANNELS = [32, 64, 128]
CNN_DROPOUT  = 0.12

# LSTM architecture
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.12

# ResNet1D architecture
RESNET_CHANNELS = [64, 128, 256]
RESNET_DROPOUT  = 0.12


# Meta-learning (MAML / Meta-SGD)
META_N_WAY         = 5       # classes per episode
META_K_SHOT        = 5       # support examples per class
META_Q_QUERY       = 10      # query examples per class per episode
META_INNER_LR      = 0.01    # inner loop learning rate
META_OUTER_LR      = 1e-3    # outer loop (meta) learning rate
META_INNER_STEPS   = 5       # gradient steps in inner loop
META_EPISODES      = 500     # training episodes
META_EVAL_EPISODES = 200     # evaluation episodes


# Continual learning (FBCL)
FBCL_EPOCHS_PER_TASK = 30
FBCL_LR              = 1e-3
FBCL_DISTILL_WEIGHT  = 0.5   # weight for knowledge distillation loss
FBCL_MEMORY_SIZE     = 50    # exemplars per class kept in memory buffer

# Noise robustness test sigmas
NOISE_SIGMAS = [0.01, 0.05, 0.10]


# Saved artifact naming conventions
SKLEARN_MODEL_FILES = {
    "rf": "rf_model.pkl",
    "xgb": "xgb_model.pkl",
    "gbm": "gbm_model.pkl",
    "lgbm": "lgbm_model.pkl",
    "stack": "stack_model.pkl",
}

TORCH_MODEL_FILES = {
    "transformer": "fault_transformer_final.pth",
    "cnn": "cnn_model.pth",
    "lstm": "lstm_model.pth",
    "resnet": "resnet_model.pth",
    "maml": "maml_final_model.pth",
    "meta_sgd": "meta_sgd_model_improved.pth",
    "fbcl": "fbcl_model.pth",
}

TORCH_MODEL_ALIASES = {
    "transformer": ["fault_transformer_final.pth", "transformer_model.pth"],
    "cnn": ["cnn_model.pth", "2dcnn_model.pth", "cnn_final.pth"],
    "lstm": ["lstm_model.pth", "lstm_final.pth"],
    "resnet": ["resnet_model.pth", "resnet1d_model.pth"],
    "maml": ["maml_final_model.pth", "maml_model.pth"],
    "meta_sgd": ["meta_sgd_model_improved.pth", "meta_sgd.pth", "meta_sgd_model.pth"],
    "fbcl": ["fbcl_model.pth"],
}

MODEL_ALIASES = {
    "rf": ["rf_model.pkl"],
    "xgb": ["xgb_model.pkl"],
    "gbm": ["gbm_model.pkl"],
    "lgbm": ["lgbm_model.pkl"],
    "stack": ["stack_model.pkl"],
    "transformer": TORCH_MODEL_ALIASES["transformer"],
    "cnn": TORCH_MODEL_ALIASES["cnn"],
    "lstm": TORCH_MODEL_ALIASES["lstm"],
    "resnet": TORCH_MODEL_ALIASES["resnet"],
    "maml": TORCH_MODEL_ALIASES["maml"],
    "meta_sgd": TORCH_MODEL_ALIASES["meta_sgd"],
    "fbcl": TORCH_MODEL_ALIASES["fbcl"],
}

MODEL_METRICS_FILES = {
    "rf": "rf_metrics.json",
    "xgb": "xgb_metrics.json",
    "gbm": "gbm_metrics.json",
    "lgbm": "lgbm_metrics.json",
    "stack": "stack_metrics.json",
    "transformer": "transformer_metrics.json",
    "cnn": "cnn_metrics.json",
    "lstm": "lstm_metrics.json",
    "resnet": "resnet_metrics.json",
    "maml": "maml_metrics.json",
    "meta_sgd": "meta_sgd_metrics.json",
    "fbcl": "fbcl_metrics.json",
}
