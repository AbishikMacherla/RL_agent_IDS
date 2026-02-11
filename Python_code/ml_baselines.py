# Machine Learning Baselines: Random Forest and XGBoost
# For comparison with RL agents

import pandas as pd
import numpy as np
import os
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Optional: XGBoost (install with: pip install xgboost)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

# ============== HYPERPARAMETERS (Documented for Methodology) ==============
# Random Forest
RF_N_ESTIMATORS = 100       # Number of trees
RF_MAX_DEPTH = 20           # Maximum tree depth
RF_MIN_SAMPLES_SPLIT = 5    # Minimum samples to split
RF_RANDOM_STATE = 42

# XGBoost
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_GAMMA = 0               # Minimum loss reduction
XGB_RANDOM_STATE = 42
# ==========================================================================


class MLBaselines:
    """
    Machine Learning baseline models for IDS comparison.
    
    Implements:
        - Random Forest (ensemble decision trees)
        - XGBoost (gradient boosting)
    """
    
    def __init__(self, data_dir='/home/abishik/HONOURS_PROJECT/processed_data'):
        self.data_dir = data_dir
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load preprocessed ML datasets."""
        print("Loading ML datasets...")
        
        self.X_train = pd.read_parquet(os.path.join(self.data_dir, 'X_train_2017.parquet'))
        self.X_test = pd.read_parquet(os.path.join(self.data_dir, 'X_test_2017.parquet'))
        self.y_train = np.load(os.path.join(self.data_dir, 'y_train_2017.npy'))
        self.y_test = np.load(os.path.join(self.data_dir, 'y_test_2017.npy'))
        
        print(f"Train: {self.X_train.shape[0]} samples")
        print(f"Test:  {self.X_test.shape[0]} samples")
        print(f"Class distribution (Train): 0={sum(self.y_train==0)}, 1={sum(self.y_train==1)}")
        
    def train_random_forest(self):
        """Train Random Forest classifier."""
        print(f"\n{'='*60}")
        print("Training Random Forest Classifier")
        print(f"{'='*60}")
        print(f"Hyperparameters:")
        print(f"  n_estimators: {RF_N_ESTIMATORS}")
        print(f"  max_depth: {RF_MAX_DEPTH}")
        print(f"  min_samples_split: {RF_MIN_SAMPLES_SPLIT}")
        
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            random_state=RF_RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        rf.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf
        
        # Evaluate
        y_pred = rf.predict(self.X_test)
        self._evaluate_model('random_forest', self.y_test, y_pred)
        
        return rf
        
    def train_xgboost(self):
        """Train XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            print("Skipping XGBoost (not installed)")
            return None
            
        print(f"\n{'='*60}")
        print("Training XGBoost Classifier")
        print(f"{'='*60}")
        print(f"Hyperparameters:")
        print(f"  n_estimators: {XGB_N_ESTIMATORS}")
        print(f"  max_depth: {XGB_MAX_DEPTH}")
        print(f"  learning_rate: {XGB_LEARNING_RATE}")
        print(f"  gamma: {XGB_GAMMA}")
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            gamma=XGB_GAMMA,
            random_state=XGB_RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=1
        )
        
        xgb_clf.fit(self.X_train, self.y_train)
        self.models['xgboost'] = xgb_clf
        
        # Evaluate
        y_pred = xgb_clf.predict(self.X_test)
        self._evaluate_model('xgboost', self.y_test, y_pred)
        
        return xgb_clf
        
    def _evaluate_model(self, model_name, y_true, y_pred):
        """Evaluate model and store results."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        print(f"\n{model_name.upper()} Results:")
        print(f"{'='*40}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
        print(f"{'='*40}")
        
    def save_models(self, output_dir='/home/abishik/HONOURS_PROJECT/models'):
        """Save trained models."""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            path = os.path.join(output_dir, f'{name}_baseline.joblib')
            joblib.dump(model, path)
            print(f"Saved {name} to {path}")
            
    def get_comparison_table(self):
        """Return comparison DataFrame for all models."""
        rows = []
        for name, metrics in self.results.items():
            rows.append({
                'Model': name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}"
            })
        return pd.DataFrame(rows)


def run_all_baselines():
    """Run all ML baseline experiments."""
    baselines = MLBaselines()
    baselines.load_data()
    
    baselines.train_random_forest()
    baselines.train_xgboost()
    
    baselines.save_models()
    
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(baselines.get_comparison_table().to_string(index=False))
    
    return baselines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML Baselines for IDS')
    parser.add_argument('--data-dir', type=str, 
                        default='/home/abishik/HONOURS_PROJECT/processed_data',
                        help='Path to processed data directory')
    args = parser.parse_args()
    
    run_all_baselines()
