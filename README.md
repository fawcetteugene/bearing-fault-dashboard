# Bearing Fault Classification — Predictive Maintenance System

Complete implementation of a bearing fault classification system using the CWRU dataset.
Covers data preprocessing, ensemble baselines, deep learning, meta-learning, continual
learning, SHAP/LIME interpretability, and a Streamlit dashboard.

The current runtime flow is inference-first:
1. Preprocess once
2. Train the candidate models once
3. Freeze the best supported model into a production bundle
4. Use the dashboard for repeated inference without retraining

## Project Structure

```
bearing_fault_project/
├── src/
│   ├── config.py            # All hyperparameters and paths
│   ├── preprocessing.py     # Cleaning, splitting, feature engineering, augmentation
│   ├── models.py            # All model definitions (Transformer, CNN, LSTM)
│   ├── train_baselines.py   # Random Forest, XGBoost, GBM training
│   ├── train_deep.py        # CNN, LSTM, Transformer training
│   ├── train_meta.py        # MAML and Meta-SGD training
│   ├── train_continual.py   # FBCL continual learning
│   ├── evaluate.py          # Unified evaluation and metrics
│   ├── interpret.py         # SHAP, LIME, attention heatmaps
│   └── utils.py             # Seeding, logging, saving helpers
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── notebooks/
│   └── colab_runner.ipynb   # Single-file Colab notebook (runs everything)
├── data/                    # Put your CSV files here
├── models/                  # Saved model checkpoints
├── outputs/                 # Plots, metrics, reports
└── requirements.txt
```

## Quick Start (Google Colab)

Open `notebooks/colab_runner.ipynb` — it installs dependencies, uploads your CSV,
runs the full pipeline, and launches the dashboard in one go.

## Quick Start (Local)

```bash
pip install -r requirements.txt

# 1. Preprocess
python src/preprocessing.py --data data/featuretime48k2048load_1.csv

# 2. Train baselines
python src/train_baselines.py

# 3. Train deep learning models
python src/train_deep.py

# 4. Train meta-learning models
python src/train_meta.py

# 5. Train continual learning model
python src/train_continual.py

# 6. Evaluate all models
python src/evaluate.py

# 7. Generate SHAP / LIME explanations
python src/interpret.py

# 8. Freeze the production bundle
python src/package_production.py

# 9. Launch dashboard
streamlit run dashboard/app.py
```

## Dataset

CWRU Bearing Dataset — `featuretime48k2048load_1.csv`
- 2,300 samples, 9 statistical features, 10 fault classes, perfectly balanced
- Source: https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets

## Target Metrics

| Metric | Target |
|--------|--------|
| Test Accuracy | ≥ 95% |
| False Positive Rate | ≤ 5% |
| Inference Time | ≤ 150 ms |
