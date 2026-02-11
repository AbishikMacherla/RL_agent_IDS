#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for RL-Enhanced IDS
Evaluates all models across standard and zero-day scenarios.
Generates results tables for the dissertation.
"""

import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Paths
DATA_DIR = '/home/abishik/HONOURS_PROJECT/processed_data'
MODEL_DIR = '/home/abishik/HONOURS_PROJECT/models'
RESULTS_DIR = '/home/abishik/HONOURS_PROJECT/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add Python_code to path for DQN import
sys.path.insert(0, '/home/abishik/HONOURS_PROJECT/Python_code')
from train_rl_agent import DQN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_data():
    """Load test data and raw labels."""
    print("Loading data...")
    
    X_test = pd.read_parquet(os.path.join(DATA_DIR, 'X_test_2017.parquet')).to_numpy()
    y_test = np.load(os.path.join(DATA_DIR, 'y_test_2017.npy'))
    
    # Full dataset for zero-day evaluation
    X_full = pd.read_parquet(os.path.join(DATA_DIR, 'X_2017_full.parquet')).to_numpy()
    y_full = np.load(os.path.join(DATA_DIR, 'y_2017_binary.npy'))
    y_labels = pd.read_csv(os.path.join(DATA_DIR, 'y_2017_labels.csv')).iloc[:, 0].values
    
    print(f"  Test set: {X_test.shape}")
    print(f"  Full dataset: {X_full.shape}")
    
    return X_test, y_test, X_full, y_full, y_labels


def evaluate_ml_model(name, model_path, X_test, y_test):
    """Evaluate a scikit-learn model."""
    if not os.path.exists(model_path):
        print(f"  [SKIP] {name}: model not found at {model_path}")
        return None
    
    print(f"  Evaluating {name}...")
    model = joblib.load(model_path)
    
    # Handle feature mismatch - ML models may have been trained on different features
    try:
        y_pred = model.predict(X_test)
    except ValueError as e:
        print(f"  [ERROR] {name}: {e}")
        return None
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'Accuracy': round(acc * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall': round(rec * 100, 2),
        'F1': round(f1 * 100, 2),
        'ConfusionMatrix': cm.tolist()
    }


def evaluate_dqn(name, model_path, X_test, y_test, state_size):
    """Evaluate a DQN model."""
    if not os.path.exists(model_path):
        print(f"  [SKIP] {name}: model not found at {model_path}")
        return None
    
    print(f"  Evaluating {name}...")
    qnetwork = DQN(state_size, action_size=2, seed=42).to(device)
    qnetwork.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    qnetwork.eval()
    
    predictions = []
    X_tensor = torch.from_numpy(X_test).float().to(device)
    
    batch_size = 2048
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            output = qnetwork(batch)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            predictions.extend(preds)
    
    y_pred = np.array(predictions)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'Accuracy': round(acc * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall': round(rec * 100, 2),
        'F1': round(f1 * 100, 2),
        'ConfusionMatrix': cm.tolist()
    }


def evaluate_ppo(name, model_path, X_test, y_test):
    """Evaluate a PPO model (Stable-Baselines3)."""
    if not os.path.exists(model_path):
        print(f"  [SKIP] {name}: model not found at {model_path}")
        return None
    
    print(f"  Evaluating {name}...")
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        
        predictions = []
        for i in range(len(X_test)):
            obs = X_test[i].astype(np.float32)
            action, _ = model.predict(obs, deterministic=True)
            predictions.append(action)
        
        y_pred = np.array(predictions)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'Accuracy': round(acc * 100, 2),
            'Precision': round(prec * 100, 2),
            'Recall': round(rec * 100, 2),
            'F1': round(f1 * 100, 2),
            'ConfusionMatrix': cm.tolist()
        }
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return None


def scenario_standard(X_test, y_test, state_size):
    """Scenario 1: Standard classification on test set."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: Standard Classification Performance")
    print("=" * 70)
    
    results = {}
    
    # ML Baselines
    for name, filename in [
        ('Random Forest', 'random_forest_baseline.joblib'),
        ('XGBoost', 'xgboost_baseline.joblib'),
    ]:
        r = evaluate_ml_model(name, os.path.join(MODEL_DIR, filename), X_test, y_test)
        if r:
            results[name] = r
    
    # Also try alternative RF filename
    if 'Random Forest' not in results:
        r = evaluate_ml_model('Random Forest', os.path.join(MODEL_DIR, 'baseline_random_forest.joblib'), X_test, y_test)
        if r:
            results['Random Forest'] = r
    
    # DQN
    r = evaluate_dqn('DQN', os.path.join(MODEL_DIR, 'dqn_agent.pth'), X_test, y_test, state_size)
    if r:
        results['DQN'] = r
    
    # PPO (commented out - focusing on DQN, RF, XGBoost for now)
    # r = evaluate_ppo('PPO', os.path.join(MODEL_DIR, 'ppo_agent.zip'), X_test, y_test)
    # if r:
    #     results['PPO'] = r
    
    return results


def scenario_zeroday_ddos(X_full, y_full, y_labels, state_size):
    """Scenario 2: Zero-Day DDoS Detection."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Zero-Day DDoS Detection")
    print("=" * 70)
    
    # Get DDoS-only samples for testing
    ddos_mask = np.char.lower(y_labels.astype(str)) == 'ddos'
    X_ddos = X_full[ddos_mask]
    y_ddos = y_full[ddos_mask]
    
    print(f"  DDoS test samples: {len(X_ddos)}")
    print(f"  DDoS attack ratio: {y_ddos.mean():.2%}")
    
    results = {}
    
    # DQN trained WITHOUT DDoS
    r = evaluate_dqn('DQN (No DDoS Train)', os.path.join(MODEL_DIR, 'dqn_no_ddos.pth'), 
                     X_ddos, y_ddos, state_size)
    if r:
        results['DQN (No DDoS)'] = r
    
    # Standard DQN for comparison (trained WITH DDoS)
    r = evaluate_dqn('DQN (Standard)', os.path.join(MODEL_DIR, 'dqn_agent.pth'), 
                     X_ddos, y_ddos, state_size)
    if r:
        results['DQN (Standard)'] = r
    
    # ML Baselines on DDoS (trained on full data including DDoS for comparison)
    for name, filename in [
        ('Random Forest', 'random_forest_baseline.joblib'),
        ('XGBoost', 'xgboost_baseline.joblib'),
    ]:
        r = evaluate_ml_model(name, os.path.join(MODEL_DIR, filename), X_ddos, y_ddos)
        if r:
            results[name] = r
    
    if 'Random Forest' not in results:
        r = evaluate_ml_model('Random Forest', os.path.join(MODEL_DIR, 'baseline_random_forest.joblib'), X_ddos, y_ddos)
        if r:
            results['Random Forest'] = r
    
    return results


def print_results_table(results, title):
    """Print formatted results table."""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─' * 65}")
    for name, metrics in results.items():
        print(f"  {name:<25} {metrics['Accuracy']:>9.2f}% {metrics['Precision']:>9.2f}% {metrics['Recall']:>9.2f}% {metrics['F1']:>9.2f}%")
    print(f"{'─' * 70}")
    
    # Print confusion matrices
    for name, metrics in results.items():
        cm = np.array(metrics['ConfusionMatrix'])
        print(f"\n  {name} Confusion Matrix:")
        print(f"    TN={cm[0][0]:>8,}  FP={cm[0][1]:>8,}")
        print(f"    FN={cm[1][0]:>8,}  TP={cm[1][1]:>8,}")


def main():
    print("=" * 70)
    print("  RL-Enhanced IDS: Comprehensive Model Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    X_test, y_test, X_full, y_full, y_labels = load_data()
    state_size = X_test.shape[1]
    print(f"  Feature dimensions: {state_size}")
    
    all_results = {}
    
    # Scenario 1: Standard
    results_standard = scenario_standard(X_test, y_test, state_size)
    print_results_table(results_standard, "Standard Classification Results")
    all_results['standard'] = results_standard
    
    # Scenario 2: Zero-Day DDoS (commented out temporarily for supervisor meeting)
    # results_zeroday = scenario_zeroday_ddos(X_full, y_full, y_labels, state_size)
    # print_results_table(results_zeroday, "Zero-Day DDoS Detection Results")
    # all_results['zeroday_ddos'] = results_zeroday
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    
    # Convert numpy types to python types for JSON serialization
    serializable = {}
    for scenario_name, scenario_results in all_results.items():
        serializable[scenario_name] = {}
        for model_name, metrics in scenario_results.items():
            serializable[scenario_name][model_name] = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in metrics.items()
            }
    
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\n\nResults saved to: {results_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
