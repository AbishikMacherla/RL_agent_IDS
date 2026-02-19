#!/usr/bin/env python3
"""
Master Experiment Script: All 4 Dissertation Scenarios
Trains and evaluates DQN, PPO, Random Forest, and XGBoost across:
  - Scenario 1: Standard Classification (78 features, CIC-IDS2017)
  - Scenario 2: Zero-Day DDoS (exclude DDoS from RL training)
  - Scenario 3: Zero-Day Web Attacks (exclude web attacks from RL training)
  - Scenario 4: Cross-Dataset Generalisation (train 2017 → test 2023, 12 features)

Usage:
    python run_all_scenarios.py                # Run all scenarios
    python run_all_scenarios.py --scenario 1   # Run specific scenario (1-4)
    python run_all_scenarios.py --eval-only    # Evaluate existing models only
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import joblib
import time
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

sys.path.insert(0, '/home/abishik/HONOURS_PROJECT/Python_code')
from train_rl_agent import train_dqn, DQN
from train_ppo_agent import train_ppo

# Paths
DATA_DIR = '/home/abishik/HONOURS_PROJECT/processed_data'
MODEL_DIR = '/home/abishik/HONOURS_PROJECT/models'
RESULTS_DIR = '/home/abishik/HONOURS_PROJECT/results'
NOTES_DIR = '/home/abishik/HONOURS_PROJECT/Notes'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Best DQN config from hyperparameter experiments (Exp 7)
BEST_DQN_CONFIG = {
    'reward_config': {'tp': 1.0, 'tn': 1.0, 'fn': -1.0, 'fp': -1.0},  # Symmetric
    'fc1_units': 64,
    'fc2_units': 64,
    'learning_rate': 5e-4,
    'episodes': 5000,
}

# Web attack labels in CIC-IDS2017 (lowercased)
WEB_ATTACK_LABELS = [
    'web attack \x96 brute force',
    'web attack \x96 sql injection', 
    'web attack \x96 xss',
    'web attack – brute force',
    'web attack – sql injection',
    'web attack – xss',
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_standard_data():
    """Load standard 2017 test data."""
    print("\n[DATA] Loading standard 2017 test data...")
    X_test = pd.read_parquet(os.path.join(DATA_DIR, 'X_test_2017.parquet')).to_numpy()
    y_test = np.load(os.path.join(DATA_DIR, 'y_test_2017.npy'))
    print(f"  Test set: {X_test.shape}, attack ratio: {y_test.mean():.2%}")
    return X_test, y_test


def load_full_data():
    """Load full 2017 dataset with string labels for zero-day filtering."""
    print("\n[DATA] Loading full 2017 dataset with labels...")
    X_full = pd.read_parquet(os.path.join(DATA_DIR, 'X_2017_full.parquet')).to_numpy()
    y_full = np.load(os.path.join(DATA_DIR, 'y_2017_binary.npy'))
    y_labels = pd.read_csv(os.path.join(DATA_DIR, 'y_2017_labels.csv')).iloc[:, 0].values
    print(f"  Full dataset: {X_full.shape}")
    
    # Print label distribution
    unique, counts = np.unique(y_labels, return_counts=True)
    print("  Label distribution:")
    for label, count in zip(unique, counts):
        print(f"    {label}: {count:,}")
    
    return X_full, y_full, y_labels


def load_cross_dataset_data():
    """Load cross-dataset generalisation data (12 common features)."""
    print("\n[DATA] Loading cross-dataset data (2017 train → 2023 test)...")
    X_train_gen = pd.read_parquet(os.path.join(DATA_DIR, 'X_gen_train_2017.parquet'))
    y_train_gen = np.load(os.path.join(DATA_DIR, 'y_gen_train_2017.npy'))
    X_test_gen = pd.read_parquet(os.path.join(DATA_DIR, 'X_gen_test_2023.parquet'))
    y_test_gen = np.load(os.path.join(DATA_DIR, 'y_gen_test_2023.npy'))
    
    print(f"  2017 train: {X_train_gen.shape}, attack ratio: {y_train_gen.mean():.2%}")
    print(f"  2023 test:  {X_test_gen.shape}, attack ratio: {y_test_gen.mean():.2%}")
    
    return X_train_gen, y_train_gen, X_test_gen, y_test_gen


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_dqn_model(name, model_path, X_test, y_test, state_size, fc1=64, fc2=64):
    """Evaluate a DQN model on test data."""
    if not os.path.exists(model_path):
        print(f"  [SKIP] {name}: model not found at {model_path}")
        return None
    
    print(f"  Evaluating {name}...")
    qnetwork = DQN(state_size, action_size=2, seed=42, fc1_units=fc1, fc2_units=fc2).to(device)
    qnetwork.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    qnetwork.eval()
    
    predictions = []
    confidence_scores = []  # Q-value difference as confidence
    X_tensor = torch.from_numpy(X_test).float().to(device)
    
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, len(X_tensor), 2048):
            batch = X_tensor[i:i+2048]
            output = qnetwork(batch)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            # Use softmax of Q-values as probability for ROC-AUC
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            predictions.extend(preds)
            confidence_scores.extend(probs)
    elapsed = time.time() - start_time
    
    return compute_metrics(name, y_test, np.array(predictions), 
                          np.array(confidence_scores), elapsed)


def evaluate_ppo_model(name, model_path, X_test, y_test):
    """Evaluate a PPO model on test data."""
    if not os.path.exists(model_path):
        print(f"  [SKIP] {name}: model not found at {model_path}")
        return None
    
    print(f"  Evaluating {name}...")
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        
        predictions = []
        confidence_scores = []
        
        start_time = time.time()
        for i in range(0, len(X_test), 1000):
            batch = X_test[i:i+1000].astype(np.float32)
            for obs in batch:
                action, _ = model.predict(obs, deterministic=True)
                predictions.append(action)
                # Get action probabilities for ROC-AUC
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                dist = model.policy.get_distribution(obs_tensor)
                prob = dist.distribution.probs[0][1].item()
                confidence_scores.append(prob)
        elapsed = time.time() - start_time
        
        return compute_metrics(name, y_test, np.array(predictions),
                              np.array(confidence_scores), elapsed)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return None


def evaluate_ml_model(name, model_path, X_test, y_test):
    """Evaluate a scikit-learn model on test data."""
    if not os.path.exists(model_path):
        print(f"  [SKIP] {name}: model not found at {model_path}")
        return None
    
    print(f"  Evaluating {name}...")
    try:
        model = joblib.load(model_path)
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        # Get probability scores for ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
        elapsed = time.time() - start_time
        
        return compute_metrics(name, y_test, y_pred, y_prob, elapsed)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return None


def compute_metrics(name, y_test, y_pred, y_prob=None, elapsed=None):
    """Compute all evaluation metrics including ROC-AUC and latency."""
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC-AUC (needs probability scores)
    auc = None
    if y_prob is not None:
        try:
            auc = round(roc_auc_score(y_test, y_prob) * 100, 2)
        except ValueError:
            auc = None  # Can fail if only one class present
    
    # Latency metrics
    n_samples = len(y_test)
    latency_us = round((elapsed / n_samples) * 1e6, 2) if elapsed else None  # microseconds per sample
    throughput = round(n_samples / elapsed) if elapsed else None  # samples per second
    
    result = {
        'Accuracy': round(acc * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall': round(rec * 100, 2),
        'F1': round(f1 * 100, 2),
        'ROC_AUC': auc,
        'Latency_us': latency_us,
        'Throughput': throughput,
        'ConfusionMatrix': cm.tolist(),
        'TN': int(cm[0][0]),
        'FP': int(cm[0][1]),
        'FN': int(cm[1][0]),
        'TP': int(cm[1][1]),
    }
    
    auc_str = f", AUC={auc}%" if auc else ""
    lat_str = f", {latency_us}μs/sample" if latency_us else ""
    print(f"    Acc={result['Accuracy']}%, Prec={result['Precision']}%, Rec={result['Recall']}%, F1={result['F1']}%{auc_str}{lat_str}")
    return result


# ============================================================================
# SCENARIO 1: STANDARD CLASSIFICATION
# ============================================================================

def scenario_1_standard(train=True, eval_only=False):
    """Standard classification on CIC-IDS2017 test set."""
    print("\n" + "#" * 70)
    print("# SCENARIO 1: STANDARD CLASSIFICATION (CIC-IDS2017)")
    print("#" * 70)
    
    X_test, y_test = load_standard_data()
    state_size = X_test.shape[1]
    results = {}
    
    if train and not eval_only:
        # Train DQN with best config (Exp 7: symmetric, 5000 episodes)
        print("\n[TRAIN] DQN (Best Config: Symmetric, 5000 eps)...")
        train_dqn(
            n_episodes=BEST_DQN_CONFIG['episodes'],
            model_name='dqn_standard.pth',
            reward_config=BEST_DQN_CONFIG['reward_config'],
            fc1_units=BEST_DQN_CONFIG['fc1_units'],
            fc2_units=BEST_DQN_CONFIG['fc2_units'],
            learning_rate=BEST_DQN_CONFIG['learning_rate'],
        )
        
        # Train PPO standard (1M timesteps for proper convergence)
        print("\n[TRAIN] PPO (1M timesteps)...")
        train_ppo(total_timesteps=1000000, model_name='ppo_standard')
        
        # Train ML baselines
        print("\n[TRAIN] ML Baselines...")
        X_train = pd.read_parquet(os.path.join(DATA_DIR, 'X_train_2017.parquet')).to_numpy()
        y_train = np.load(os.path.join(DATA_DIR, 'y_train_2017.npy'))
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest_baseline.joblib'))
        print("  RF trained and saved.")
        
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgboost_baseline.joblib'))
        print("  XGBoost trained and saved.")
    
    # Evaluate all
    r = evaluate_ml_model('Random Forest', os.path.join(MODEL_DIR, 'random_forest_baseline.joblib'), X_test, y_test)
    if r: results['Random Forest'] = r
    
    r = evaluate_ml_model('XGBoost', os.path.join(MODEL_DIR, 'xgboost_baseline.joblib'), X_test, y_test)
    if r: results['XGBoost'] = r
    
    r = evaluate_dqn_model('DQN', os.path.join(MODEL_DIR, 'dqn_standard.pth'), X_test, y_test, state_size)
    if not r:
        r = evaluate_dqn_model('DQN', os.path.join(MODEL_DIR, 'dqn_agent.pth'), X_test, y_test, state_size)
    if r: results['DQN'] = r
    
    r = evaluate_ppo_model('PPO', os.path.join(MODEL_DIR, 'ppo_standard.zip'), X_test, y_test)
    if not r:
        r = evaluate_ppo_model('PPO', os.path.join(MODEL_DIR, 'ppo_agent.zip'), X_test, y_test)
    if r: results['PPO'] = r
    
    return results


# ============================================================================
# SCENARIO 2: ZERO-DAY DDoS
# ============================================================================

def scenario_2_zeroday_ddos(train=True, eval_only=False):
    """Train without DDoS, test on DDoS-only samples."""
    print("\n" + "#" * 70)
    print("# SCENARIO 2: ZERO-DAY DDoS DETECTION")
    print("#" * 70)
    
    X_full, y_full, y_labels = load_full_data()
    state_size = X_full.shape[1]
    
    # Identify DDoS labels
    ddos_labels = [l for l in np.unique(y_labels) if 'ddos' in l.lower()]
    print(f"\n  DDoS labels found: {ddos_labels}")
    
    # Get DDoS-only test set
    ddos_mask = np.isin(y_labels, ddos_labels)
    X_ddos = X_full[ddos_mask]
    y_ddos = y_full[ddos_mask]
    print(f"  DDoS test samples: {len(X_ddos)}, attack ratio: {y_ddos.mean():.2%}")
    
    if train and not eval_only:
        # Train DQN excluding DDoS
        print("\n[TRAIN] DQN (No DDoS)...")
        train_dqn(
            exclude_labels=ddos_labels,
            n_episodes=BEST_DQN_CONFIG['episodes'],
            model_name='dqn_no_ddos.pth',
            reward_config=BEST_DQN_CONFIG['reward_config'],
            fc1_units=BEST_DQN_CONFIG['fc1_units'],
            fc2_units=BEST_DQN_CONFIG['fc2_units'],
            learning_rate=BEST_DQN_CONFIG['learning_rate'],
        )
        
        # Train PPO excluding DDoS (1M timesteps)
        print("\n[TRAIN] PPO (No DDoS)...")
        train_ppo(
            exclude_labels=ddos_labels,
            total_timesteps=1000000,
            model_name='ppo_no_ddos',
        )
    
    results = {}
    
    # RL models trained WITHOUT DDoS
    r = evaluate_dqn_model('DQN (No DDoS)', os.path.join(MODEL_DIR, 'dqn_no_ddos.pth'), X_ddos, y_ddos, state_size)
    if r: results['DQN (No DDoS)'] = r
    
    r = evaluate_ppo_model('PPO (No DDoS)', os.path.join(MODEL_DIR, 'ppo_no_ddos.zip'), X_ddos, y_ddos)
    if r: results['PPO (No DDoS)'] = r
    
    # Standard models (trained WITH DDoS — for comparison)
    r = evaluate_dqn_model('DQN (Standard)', os.path.join(MODEL_DIR, 'dqn_standard.pth'), X_ddos, y_ddos, state_size)
    if not r:
        r = evaluate_dqn_model('DQN (Standard)', os.path.join(MODEL_DIR, 'dqn_agent.pth'), X_ddos, y_ddos, state_size)
    if r: results['DQN (Standard)'] = r
    
    r = evaluate_ml_model('Random Forest', os.path.join(MODEL_DIR, 'random_forest_baseline.joblib'), X_ddos, y_ddos)
    if r: results['Random Forest'] = r
    
    r = evaluate_ml_model('XGBoost', os.path.join(MODEL_DIR, 'xgboost_baseline.joblib'), X_ddos, y_ddos)
    if r: results['XGBoost'] = r
    
    return results


# ============================================================================
# SCENARIO 3: ZERO-DAY WEB ATTACKS
# ============================================================================

def scenario_3_zeroday_web(train=True, eval_only=False):
    """Train without web attacks, test on web-attack-only samples."""
    print("\n" + "#" * 70)
    print("# SCENARIO 3: ZERO-DAY WEB ATTACK DETECTION")
    print("#" * 70)
    
    X_full, y_full, y_labels = load_full_data()
    state_size = X_full.shape[1]
    
    # Identify web attack labels
    web_labels = [l for l in np.unique(y_labels) 
                  if 'web' in l.lower() or 'sql' in l.lower() or 'xss' in l.lower()]
    print(f"\n  Web attack labels found: {web_labels}")
    
    if not web_labels:
        # Try broader matching
        web_labels = [l for l in np.unique(y_labels) if 'web' in l.lower()]
        print(f"  Fallback web labels: {web_labels}")
    
    if not web_labels:
        print("  ERROR: No web attack labels found! Skipping scenario.")
        return {}
    
    # Get web-attack-only test set
    web_mask = np.isin(y_labels, web_labels)
    X_web = X_full[web_mask]
    y_web = y_full[web_mask]
    print(f"  Web attack test samples: {len(X_web)}, attack ratio: {y_web.mean():.2%}")
    
    if train and not eval_only:
        # Train DQN excluding web attacks
        print("\n[TRAIN] DQN (No Web Attacks)...")
        train_dqn(
            exclude_labels=web_labels,
            n_episodes=BEST_DQN_CONFIG['episodes'],
            model_name='dqn_no_web.pth',
            reward_config=BEST_DQN_CONFIG['reward_config'],
            fc1_units=BEST_DQN_CONFIG['fc1_units'],
            fc2_units=BEST_DQN_CONFIG['fc2_units'],
            learning_rate=BEST_DQN_CONFIG['learning_rate'],
        )
        
        # Train PPO excluding web attacks (1M timesteps)
        print("\n[TRAIN] PPO (No Web Attacks)...")
        train_ppo(
            exclude_labels=web_labels,
            total_timesteps=1000000,
            model_name='ppo_no_web',
        )
    
    results = {}
    
    # RL models trained WITHOUT web attacks
    r = evaluate_dqn_model('DQN (No Web)', os.path.join(MODEL_DIR, 'dqn_no_web.pth'), X_web, y_web, state_size)
    if r: results['DQN (No Web)'] = r
    
    r = evaluate_ppo_model('PPO (No Web)', os.path.join(MODEL_DIR, 'ppo_no_web.zip'), X_web, y_web)
    if r: results['PPO (No Web)'] = r
    
    # Standard models for comparison
    r = evaluate_dqn_model('DQN (Standard)', os.path.join(MODEL_DIR, 'dqn_standard.pth'), X_web, y_web, state_size)
    if not r:
        r = evaluate_dqn_model('DQN (Standard)', os.path.join(MODEL_DIR, 'dqn_agent.pth'), X_web, y_web, state_size)
    if r: results['DQN (Standard)'] = r
    
    r = evaluate_ml_model('Random Forest', os.path.join(MODEL_DIR, 'random_forest_baseline.joblib'), X_web, y_web)
    if r: results['Random Forest'] = r
    
    r = evaluate_ml_model('XGBoost', os.path.join(MODEL_DIR, 'xgboost_baseline.joblib'), X_web, y_web)
    if r: results['XGBoost'] = r
    
    return results


# ============================================================================
# SCENARIO 4: CROSS-DATASET GENERALISATION (2017 → 2023)
# ============================================================================

def scenario_4_cross_dataset(train=True, eval_only=False):
    """Train on 2017 common features, test on 2023 IoT dataset."""
    print("\n" + "#" * 70)
    print("# SCENARIO 4: CROSS-DATASET GENERALISATION (2017 → 2023)")
    print("#" * 70)
    
    X_train_gen, y_train_gen, X_test_gen, y_test_gen = load_cross_dataset_data()
    state_size = X_train_gen.shape[1]  # 12 features
    
    print(f"  Feature count: {state_size} (common features only)")
    
    if train and not eval_only:
        # Train DQN on 12-feature data
        print("\n[TRAIN] DQN (12 features, 2017 data)...")
        # Need a custom IdsEnv that loads the 12-feature data
        # Save temp files for the env
        temp_X_path = os.path.join(DATA_DIR, 'X_gen_full_2017.parquet')
        temp_y_path = os.path.join(DATA_DIR, 'y_gen_binary_2017.npy')
        
        if not os.path.exists(temp_X_path):
            X_train_gen.to_parquet(temp_X_path, index=False) if isinstance(X_train_gen, pd.DataFrame) else pd.DataFrame(X_train_gen).to_parquet(temp_X_path, index=False)
            np.save(temp_y_path, y_train_gen)
        
        # Train DQN directly on the gen data
        train_dqn_on_data(
            X_train_gen.to_numpy() if isinstance(X_train_gen, pd.DataFrame) else X_train_gen,
            y_train_gen,
            model_name='dqn_cross_dataset.pth',
            episodes=BEST_DQN_CONFIG['episodes'],
            reward_config=BEST_DQN_CONFIG['reward_config'],
            fc1_units=BEST_DQN_CONFIG['fc1_units'],
            fc2_units=BEST_DQN_CONFIG['fc2_units'],
            learning_rate=BEST_DQN_CONFIG['learning_rate'],
        )
        
        # Train PPO on 12-feature data (1M timesteps)
        print("\n[TRAIN] PPO (12 features, 2017 data, 1M timesteps)...")
        train_ppo_on_data(
            X_train_gen.to_numpy() if isinstance(X_train_gen, pd.DataFrame) else X_train_gen,
            y_train_gen,
            model_name='ppo_cross_dataset',
            timesteps=1000000,
        )
        
        # Train ML baselines on 12-feature data
        print("\n[TRAIN] ML Baselines (12 features)...")
        X_train_np = X_train_gen.to_numpy() if isinstance(X_train_gen, pd.DataFrame) else X_train_gen
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
        rf.fit(X_train_np, y_train_gen)
        joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_cross_dataset.joblib'))
        print("  RF (12-feat) trained and saved.")
        
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train_np, y_train_gen)
        joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_cross_dataset.joblib'))
        print("  XGBoost (12-feat) trained and saved.")
    
    # Evaluate all on 2023 data
    X_test_np = X_test_gen.to_numpy() if isinstance(X_test_gen, pd.DataFrame) else X_test_gen
    results = {}
    
    r = evaluate_ml_model('Random Forest', os.path.join(MODEL_DIR, 'rf_cross_dataset.joblib'), X_test_np, y_test_gen)
    if r: results['Random Forest'] = r
    
    r = evaluate_ml_model('XGBoost', os.path.join(MODEL_DIR, 'xgb_cross_dataset.joblib'), X_test_np, y_test_gen)
    if r: results['XGBoost'] = r
    
    r = evaluate_dqn_model('DQN', os.path.join(MODEL_DIR, 'dqn_cross_dataset.pth'), X_test_np, y_test_gen, state_size)
    if r: results['DQN'] = r
    
    r = evaluate_ppo_model('PPO', os.path.join(MODEL_DIR, 'ppo_cross_dataset.zip'), X_test_np, y_test_gen)
    if r: results['PPO'] = r
    
    return results


# ============================================================================
# CUSTOM TRAINING FUNCTIONS (for cross-dataset with different feature count)
# ============================================================================

def train_dqn_on_data(X, y, model_name, episodes, reward_config, fc1_units, fc2_units, learning_rate):
    """Train DQN directly on provided data (bypasses IdsEnv file loading)."""
    import torch.nn.functional as F
    import torch.optim as optim
    from collections import deque, namedtuple
    import random
    
    state_size = X.shape[1]
    action_size = 2
    rewards = reward_config or {'tp': 10.0, 'tn': 1.0, 'fn': -10.0, 'fp': -1.0}
    
    print(f"\n  Training DQN on {X.shape[0]} samples, {state_size} features")
    print(f"  Episodes: {episodes}, Network: {fc1_units}-{fc2_units}, LR: {learning_rate}")
    print(f"  Rewards: {rewards}")
    
    # Build network
    qnetwork_local = DQN(state_size, action_size, seed=42, fc1_units=fc1_units, fc2_units=fc2_units).to(device)
    qnetwork_target = DQN(state_size, action_size, seed=42, fc1_units=fc1_units, fc2_units=fc2_units).to(device)
    optimizer = optim.Adam(qnetwork_local.parameters(), lr=learning_rate)
    
    # Replay buffer
    Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])
    memory = deque(maxlen=100000)
    batch_size = 64
    gamma = 0.99
    tau = 1e-3
    
    scores = []
    scores_window = deque(maxlen=100)
    eps = 1.0
    max_t = 1000
    
    for i_episode in range(1, episodes + 1):
        # Random start
        max_start = max(0, len(X) - 2000)
        idx = random.randint(0, max_start) if max_start > 0 else 0
        score = 0
        
        for t in range(max_t):
            if idx >= len(X):
                break
            
            state = X[idx].astype(np.float32)
            true_label = y[idx]
            
            # Epsilon-greedy action
            if random.random() > eps:
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                qnetwork_local.eval()
                with torch.no_grad():
                    action_values = qnetwork_local(state_t)
                qnetwork_local.train()
                action = np.argmax(action_values.cpu().data.numpy())
            else:
                action = random.choice([0, 1])
            
            # Reward
            if action == true_label:
                reward = rewards['tp'] if action == 1 else rewards['tn']
            else:
                reward = rewards['fn'] if true_label == 1 else rewards['fp']
            
            idx += 1
            done = idx >= len(X)
            next_state = X[idx].astype(np.float32) if not done else np.zeros(state_size, dtype=np.float32)
            
            # Store experience
            memory.append(Experience(state, action, reward, next_state, done))
            
            # Learn
            if len(memory) > batch_size and t % 4 == 0:
                experiences = random.sample(list(memory), k=batch_size)
                states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
                actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
                r_batch = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
                next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
                dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
                
                Q_targets_next = qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
                Q_targets = r_batch + (gamma * Q_targets_next * (1 - dones))
                Q_expected = qnetwork_local(states).gather(1, actions)
                
                loss = F.mse_loss(Q_expected, Q_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Soft update
                for target_param, local_param in zip(qnetwork_target.parameters(), qnetwork_local.parameters()):
                    target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
            score += reward
            if done:
                break
        
        scores_window.append(score)
        scores.append(score)
        eps = max(0.01, 0.999 * eps)
        
        print(f'\r  Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\r  Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
    
    model_path = os.path.join(MODEL_DIR, model_name)
    torch.save(qnetwork_local.state_dict(), model_path)
    print(f"\n  Model saved to {model_path}")


def train_ppo_on_data(X, y, model_name, timesteps):
    """Train PPO directly on provided data using a custom Gym env."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    class SimpleIdsEnv(gym.Env):
        """Minimal IDS env that works directly with provided data arrays."""
        metadata = {'render_modes': ['human']}
        
        def __init__(self, X_data, y_data, reward_config=None):
            super().__init__()
            self.X = X_data
            self.y = y_data
            self.rewards = reward_config or {'tp': 10.0, 'tn': 1.0, 'fn': -10.0, 'fp': -1.0}
            self.current_step = 0
            self.max_steps = len(self.X) - 1
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.X.shape[1],), dtype=np.float32)
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            max_start = max(0, len(self.X) - 2000)
            self.current_step = np.random.randint(0, max_start) if max_start > 0 else 0
            return self.X[self.current_step].astype(np.float32), {}
        
        def step(self, action):
            true_label = self.y[self.current_step]
            if action == true_label:
                reward = self.rewards['tp'] if action == 1 else self.rewards['tn']
            else:
                reward = self.rewards['fn'] if true_label == 1 else self.rewards['fp']
            self.current_step += 1
            terminated = self.current_step >= self.max_steps
            obs = self.X[self.current_step].astype(np.float32) if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, reward, terminated, False, {}
    
    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    
    print(f"\n  Training PPO on {X_np.shape[0]} samples, {X_np.shape[1]} features")
    print(f"  Timesteps: {timesteps}")
    
    env = DummyVecEnv([lambda: SimpleIdsEnv(X_np, y)])
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )
    
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"\n  Model saved to {model_path}.zip")


# ============================================================================
# REPORTING
# ============================================================================

def print_results_table(results, title):
    """Print formatted results table."""
    print(f"\n{'─' * 100}")
    print(f"  {title}")
    print(f"{'─' * 100}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'μs/sample':>10}")
    print(f"  {'─' * 95}")
    for name, m in results.items():
        auc = f"{m.get('ROC_AUC', 'N/A')}%" if m.get('ROC_AUC') else "N/A"
        lat = f"{m.get('Latency_us', 'N/A')}" if m.get('Latency_us') else "N/A"
        print(f"  {name:<25} {m['Accuracy']:>9.2f}% {m['Precision']:>9.2f}% {m['Recall']:>9.2f}% {m['F1']:>9.2f}% {auc:>10} {lat:>10}")
    print(f"{'─' * 100}")
    
    for name, m in results.items():
        cm = np.array(m['ConfusionMatrix'])
        tp_str = f"  {name}: TN={cm[0][0]:>8,}  FP={cm[0][1]:>8,}  FN={cm[1][0]:>8,}  TP={cm[1][1]:>8,}"
        if m.get('Throughput'):
            tp_str += f"  ({m['Throughput']:,} samples/sec)"
        print(f"\n{tp_str}")


def generate_scenario_note(scenario_num, title, results, description):
    """Generate a markdown note for a scenario."""
    note = f"""# Scenario {scenario_num}: {title}
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Description
{description}

## Results
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
"""
    for name, m in results.items():
        note += f"| **{name}** | {m['Accuracy']}% | {m['Precision']}% | {m['Recall']}% | {m['F1']}% |\n"
    
    note += "\n## Confusion Matrices\n"
    for name, m in results.items():
        note += f"\n### {name}\n"
        note += f"| | Predicted Allow | Predicted Block |\n"
        note += f"|--|--|--|\n"
        note += f"| **Actual Benign** | TN = {m['TN']:,} | FP = {m['FP']:,} |\n"
        note += f"| **Actual Attack** | FN = {m['FN']:,} | TP = {m['TP']:,} |\n"
    
    # Analysis
    note += "\n## Analysis\n"
    if results:
        best_f1_name = max(results, key=lambda k: results[k]['F1'])
        best_recall_name = max(results, key=lambda k: results[k]['Recall'])
        note += f"- **Best F1**: {best_f1_name} ({results[best_f1_name]['F1']}%)\n"
        note += f"- **Best Recall**: {best_recall_name} ({results[best_recall_name]['Recall']}%)\n"
    
    return note


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run all dissertation scenarios')
    parser.add_argument('--scenario', type=int, choices=[1, 2, 3, 4], help='Run specific scenario')
    parser.add_argument('--eval-only', action='store_true', help='Evaluate existing models only')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  RL-ENHANCED IDS: COMPLETE EXPERIMENT SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_results = {}
    train = not args.eval_only
    
    scenarios = {
        1: ('Standard Classification', scenario_1_standard,
            'Standard binary classification on CIC-IDS2017 test set. All models trained and tested on known attack types.'),
        2: ('Zero-Day DDoS Detection', scenario_2_zeroday_ddos,
            'RL agents trained WITHOUT DDoS attacks, then tested on DDoS-only data. Tests whether agents can detect novel high-volume threats.'),
        3: ('Zero-Day Web Attack Detection', scenario_3_zeroday_web,
            'RL agents trained WITHOUT web attacks (SQL injection, XSS, brute force), tested on web-attack-only data. Tests detection of stealthier application-layer threats.'),
        4: ('Cross-Dataset Generalisation', scenario_4_cross_dataset,
            'All models trained on CIC-IDS2017 (2017) using 12 common features, tested on CIC-IoT-2023. Tests whether models generalise across different networks and time periods.'),
    }
    
    run_scenarios = [args.scenario] if args.scenario else [1, 2, 3, 4]
    
    for s_num in run_scenarios:
        title, func, desc = scenarios[s_num]
        results = func(train=train, eval_only=args.eval_only)
        
        if results:
            print_results_table(results, f"Scenario {s_num}: {title}")
            all_results[f'scenario_{s_num}'] = results
            
            # Generate note
            note = generate_scenario_note(s_num, title, results, desc)
            note_path = os.path.join(NOTES_DIR, f'scenario_{s_num}_results.md')
            with open(note_path, 'w') as f:
                f.write(note)
            print(f"\n  Note saved to {note_path}")
    
    # Save all results
    results_path = os.path.join(RESULTS_DIR, 'all_scenarios_results.json')
    
    # Make JSON serializable
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
    
    print(f"\n\nAll results saved to: {results_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
