# ML Baseline Training Script
# Trains Random Forest, XGBoost, and Logistic Regression on the ML train split

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Optional: XGBoost (install with: pip install xgboost)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Run 'pip install xgboost' to enable.")

DATA_DIR = '/home/abishik/HONOURS_PROJECT/processed_data'
MODEL_DIR = '/home/abishik/HONOURS_PROJECT/models'
os.makedirs(MODEL_DIR, exist_ok=True)


def load_ml_data():
    """Load the ML training data (80% split)."""
    X_train_path = os.path.join(DATA_DIR, 'X_train_ml.parquet')
    y_train_path = os.path.join(DATA_DIR, 'y_train_ml.npy')
    
    # Fallback to old file names
    if not os.path.exists(X_train_path):
        print("Warning: Using legacy file names. Run data_preprocessing.py first.")
        X_train_path = os.path.join(DATA_DIR, 'X_processed.parquet')
        y_train_path = os.path.join(DATA_DIR, 'y_encoded.npy')
    
    X_train = pd.read_parquet(X_train_path).to_numpy()
    y_train = np.load(y_train_path)
    
    print(f"Loaded training data: X={X_train.shape}, y={y_train.shape}")
    return X_train, y_train


def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    model_path = os.path.join(MODEL_DIR, 'baseline_random_forest.joblib')
    joblib.dump(rf, model_path)
    print(f"Saved to {model_path}")
    return rf


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression classifier."""
    print("\n--- Training Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    
    model_path = os.path.join(MODEL_DIR, 'baseline_logistic_regression.joblib')
    joblib.dump(lr, model_path)
    print(f"Saved to {model_path}")
    return lr


def train_xgboost(X_train, y_train):
    """Train XGBoost classifier."""
    if not XGBOOST_AVAILABLE:
        print("Skipping XGBoost (not installed).")
        return None
    
    print("\n--- Training XGBoost ---")
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    model_path = os.path.join(MODEL_DIR, 'baseline_xgboost.joblib')
    joblib.dump(xgb, model_path)
    print(f"Saved to {model_path}")
    return xgb


if __name__ == '__main__':
    print("=" * 60)
    print("ML BASELINE TRAINING")
    print("=" * 60)
    
    X_train, y_train = load_ml_data()
    
    train_random_forest(X_train, y_train)
    train_logistic_regression(X_train, y_train)
    train_xgboost(X_train, y_train)
    
    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE!")
    print("=" * 60)
