# ============================================================================
# HYPERPARAMETER REFERENCE GUIDE
# ============================================================================
# This file documents all tunable parameters for the RL-Enhanced IDS project.
# Use these values as starting points. Adjust based on training results.
#
# Project Goal: Proof of concept for RL-Enhanced Intrusion Detection System
#               for Autonomous Network Defence
# ============================================================================

# ===========================================================================
# DQN (Deep Q-Network) PARAMETERS
# ===========================================================================
# DQN is a value-based RL algorithm that learns Q-values for state-action pairs.
# It uses experience replay and a target network for stable training.

DQN_PARAMS = {
    # Neural Network Architecture
    'fc1_units': 64,           # First hidden layer neurons (try 128 if underfitting)
    'fc2_units': 64,           # Second hidden layer neurons
    'activation': 'ReLU',      # Activation function
    
    # Training Parameters
    'learning_rate': 0.0005,   # Adam optimizer LR (lower if unstable: 0.0001)
    'buffer_size': 100000,     # Experience replay buffer size
    'batch_size': 64,          # Minibatch size (smaller = less bias to recent)
    'gamma': 0.99,             # Discount factor (future reward importance)
    'tau': 0.001,              # Soft update coefficient for target network
    'update_every': 4,         # Steps between network updates
    
    # Exploration (Epsilon-Greedy)
    'eps_start': 1.0,          # Initial exploration rate (fully random)
    'eps_end': 0.05,           # Final exploration rate (some randomness)
    'eps_decay': 0.999,        # Decay rate per episode
    
    # Training Duration
    'n_episodes': 2000,        # Total training episodes
    'max_steps': 1000,         # Max steps per episode
}

# Troubleshooting DQN:
# - Training unstable? Lower learning_rate to 0.0001
# - Not learning? Increase buffer_size, slow eps_decay (0.9999)
# - Overfitting? Increase batch_size, add dropout
# - Low recall? Increase TP_REWARD, increase FN_PENALTY


# ===========================================================================
# PPO (Proximal Policy Optimization) PARAMETERS
# ===========================================================================
# PPO is a policy-based RL algorithm that is generally more stable than DQN.
# It clips policy updates to prevent too large changes.

PPO_PARAMS = {
    # Policy Network (MlpPolicy in Stable-Baselines3)
    'policy': 'MlpPolicy',
    'net_arch': [64, 64],      # Hidden layer sizes
    
    # Training Parameters
    'learning_rate': 0.0003,   # Adam optimizer LR (PPO is less sensitive)
    'n_steps': 2048,           # Steps before each update
    'batch_size': 64,          # Minibatch size
    'n_epochs': 10,            # Optimization epochs per update
    'gamma': 0.99,             # Discount factor
    'gae_lambda': 0.95,        # GAE lambda for advantage estimation
    
    # PPO Specific
    'clip_range': 0.2,         # Policy clip range (prevents large updates)
    'ent_coef': 0.01,          # Entropy coefficient (higher = more exploration)
    'vf_coef': 0.5,            # Value function coefficient
    'max_grad_norm': 0.5,      # Gradient clipping
    
    # Training Duration
    'total_timesteps': 100000, # Total training timesteps
}

# Troubleshooting PPO:
# - Not exploring enough? Increase ent_coef to 0.02-0.05
# - Updates too aggressive? Lower clip_range to 0.1
# - Slow convergence? Increase n_epochs, lower batch_size


# ===========================================================================
# RANDOM FOREST PARAMETERS
# ===========================================================================
# Random Forest is an ensemble of decision trees. Strong baseline for IDS.

RF_PARAMS = {
    'n_estimators': 100,       # Number of trees (more = better, slower)
    'max_depth': 20,           # Max tree depth (higher = risk of overfit)
    'min_samples_split': 5,    # Min samples to split node
    'min_samples_leaf': 2,     # Min samples in leaf node
    'max_features': 'sqrt',    # Features per split
    'random_state': 42,        # Reproducibility
    'n_jobs': -1,              # Use all CPU cores
}


# ===========================================================================
# XGBOOST PARAMETERS
# ===========================================================================
# XGBoost is gradient boosting. Often achieves highest accuracy on IDS tasks.

XGB_PARAMS = {
    'n_estimators': 100,       # Number of boosting rounds
    'max_depth': 6,            # Max tree depth
    'learning_rate': 0.1,      # Shrinkage (lower = more trees needed)
    'gamma': 0,                # Min loss reduction to split
    'min_child_weight': 1,     # Min sum of instance weight in child
    'subsample': 0.8,          # Row sampling ratio
    'colsample_bytree': 0.8,   # Column sampling ratio
    'random_state': 42,
}


# ===========================================================================
# REWARD STRUCTURE (Security-First Approach)
# ===========================================================================
# Asymmetric rewards prioritize catching attacks over avoiding false alarms.
# This reflects real-world security where missing an attack is worse than
# a false alarm.

REWARD_PARAMS = {
    'true_positive': 10,       # Correctly block malicious traffic (HIGH)
    'true_negative': 1,        # Correctly allow benign traffic (low)
    'false_negative': -10,     # Miss malicious traffic (CRITICAL PENALTY)
    'false_positive': -1,      # Block benign traffic (minor penalty)
}

# Rationale:
# - TP >> TN: Catching attacks is the primary goal
# - FN >> FP: Missing an attack is worse than a false alarm
# - This leads to high recall at the cost of some precision


# ===========================================================================
# ENVIRONMENT PARAMETERS
# ===========================================================================
# IdsEnv configuration for the OpenAI Gym environment.

ENV_PARAMS = {
    'data_dir': '/home/abishik/HONOURS_PROJECT/processed_data',
    'feature_scaling': 'MinMax',  # Features scaled to [0, 1]
    'action_space': 2,            # 0 = Allow, 1 = Block
    'observation_space': 78,      # Number of network features
}


# ===========================================================================
# EXPERIMENT SCENARIOS
# ===========================================================================
# Different testing scenarios to evaluate the models.

SCENARIOS = {
    'standard': {
        'description': 'Train and test on same dataset (CIC-IDS2017)',
        'train_data': 'CIC-IDS2017',
        'test_data': 'CIC-IDS2017 (held-out 20%)',
        'exclude_labels': None,
    },
    'zero_day': {
        'description': 'Exclude attack type from training, test on it',
        'train_data': 'CIC-IDS2017 (minus held-out attack)',
        'test_data': 'CIC-IDS2017 (held-out attack only)',
        'exclude_labels': ['ddos'],  # Example: hold out DDoS attacks
    },
    'generalization': {
        'description': 'Train on one dataset, test on another',
        'train_data': 'CIC-IDS2017',
        'test_data': 'CIC-IoT-2023',
        'exclude_labels': None,
    },
}
