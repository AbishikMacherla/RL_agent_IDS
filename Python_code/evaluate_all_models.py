# Unified Model Evaluation Script
# Evaluates all trained models on the UNTOUCHED test set

import pandas as pd
import numpy as np
import os
import torch
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support

DATA_DIR = '/home/abishik/HONOURS_PROJECT/processed_data'
MODEL_DIR = '/home/abishik/HONOURS_PROJECT/models'

# Import DQN architecture for loading RL models
from train_rl_agent import DQN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test_data():
    """Load the UNTOUCHED test set (20% split)."""
    X_test_path = os.path.join(DATA_DIR, 'X_test_ml_untouched.parquet')
    y_test_path = os.path.join(DATA_DIR, 'y_test_ml_untouched.npy')
    
    # Fallback
    if not os.path.exists(X_test_path):
        print("Warning: Using full dataset for testing (legacy mode).")
        X_test_path = os.path.join(DATA_DIR, 'X_processed.parquet')
        y_test_path = os.path.join(DATA_DIR, 'y_encoded.npy')
    
    X_test = pd.read_parquet(X_test_path).to_numpy()
    y_test = np.load(y_test_path)
    
    print(f"Loaded test data: X={X_test.shape}, y={y_test.shape}")
    return X_test, y_test


def evaluate_ml_model(model_path, X_test, y_test):
    """Evaluate a scikit-learn model."""
    if not os.path.exists(model_path):
        return None
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}


def evaluate_dqn_model(model_path, X_test, y_test, state_size):
    """Evaluate a trained DQN model."""
    if not os.path.exists(model_path):
        return None
    
    qnetwork = DQN(state_size, action_size=2, seed=42).to(device)
    qnetwork.load_state_dict(torch.load(model_path, map_location=device))
    qnetwork.eval()
    
    X_tensor = torch.from_numpy(X_test).float().to(device)
    predictions = []
    
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            output = qnetwork(batch)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            predictions.extend(preds)
    
    y_pred = np.array(predictions)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}


def main():
    print("=" * 70)
    print("UNIFIED MODEL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 70)
    
    X_test, y_test = load_test_data()
    state_size = X_test.shape[1]
    
    results = {}
    
    # ML Models
    models_to_evaluate = [
        ('Random Forest', os.path.join(MODEL_DIR, 'baseline_random_forest.joblib')),
        ('Logistic Regression', os.path.join(MODEL_DIR, 'baseline_logistic_regression.joblib')),
        ('XGBoost', os.path.join(MODEL_DIR, 'baseline_xgboost.joblib')),
    ]
    
    for name, path in models_to_evaluate:
        result = evaluate_ml_model(path, X_test, y_test)
        if result:
            results[name] = result
            print(f"\n{name}: Acc={result['Accuracy']:.4f}, Prec={result['Precision']:.4f}, Rec={result['Recall']:.4f}, F1={result['F1']:.4f}")
    
    # DQN Models
    dqn_models = [
        ('DQN (Standard)', os.path.join(MODEL_DIR, 'dqn_agent.pth')),
        ('DQN (Zero-Day)', os.path.join(MODEL_DIR, 'dqn_agent_zeroday.pth')),
    ]
    
    for name, path in dqn_models:
        result = evaluate_dqn_model(path, X_test, y_test, state_size)
        if result:
            results[name] = result
            print(f"\n{name}: Acc={result['Accuracy']:.4f}, Prec={result['Precision']:.4f}, Rec={result['Recall']:.4f}, F1={result['F1']:.4f}")
    
    # Summary Table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['Accuracy']:<12.4f} {metrics['Precision']:<12.4f} {metrics['Recall']:<12.4f} {metrics['F1']:<12.4f}")
    
    return results


if __name__ == '__main__':
    main()
