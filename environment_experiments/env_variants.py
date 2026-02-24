"""
Alternative IDS Environments for RL experiments.
Each environment variant modifies a different aspect of the base IdsEnv.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = '/home/abishik/HONOURS_PROJECT/processed_data'


class IdsEnvFeatureSubset(gym.Env):
    """
    Experiment 1: Feature Subset Environment.
    Uses only the top-K most important features (selected via RF importance).
    Hypothesis: fewer, better features → less noise → better generalisation.
    """
    metadata = {'render_modes': []}

    def __init__(self, data_dir=DATA_DIR, exclude_labels=None, reward_config=None,
                 top_k=20):
        super().__init__()
        self.rewards = reward_config or {'tp': 1.0, 'tn': 1.0, 'fn': -1.0, 'fp': -1.0}
        self.top_k = top_k

        # Load data
        X_train = pd.read_parquet(os.path.join(data_dir, 'X_train.parquet')).values
        y_train = pd.read_parquet(os.path.join(data_dir, 'y_train.parquet')).values.ravel()
        X_test = pd.read_parquet(os.path.join(data_dir, 'X_test.parquet')).values
        y_test = pd.read_parquet(os.path.join(data_dir, 'y_test.parquet')).values.ravel()

        # Select top-K features using RF importance (quick fit on subset)
        print(f"  [Env1] Selecting top-{top_k} features via RF importance...")
        sample_idx = np.random.choice(len(X_train), min(50000, len(X_train)), replace=False)
        rf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train[sample_idx], y_train[sample_idx])
        self.feature_indices = np.argsort(rf.feature_importances_)[-top_k:]
        print(f"  [Env1] Selected feature indices: {sorted(self.feature_indices)}")

        self.X = X_train[:, self.feature_indices].astype(np.float32)
        self.y = y_train
        self.X_test = X_test[:, self.feature_indices].astype(np.float32)
        self.y_test = y_test

        # Handle label exclusion for zero-day
        if exclude_labels is not None:
            raw_labels = pd.read_parquet(os.path.join(data_dir, 'y_train.parquet'))
            if 'label_raw' in raw_labels.columns:
                mask = ~raw_labels['label_raw'].isin(exclude_labels)
                self.X = self.X[mask.values]
                self.y = self.y[mask.values]

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(top_k,), dtype=np.float32)
        self.current_idx = 0
        self.max_steps = min(1000, len(self.X))
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = np.random.randint(0, len(self.X) - self.max_steps)
        self.steps = 0
        return self.X[self.current_idx], {}

    def step(self, action):
        true_label = self.y[self.current_idx]
        if action == 1 and true_label == 1:
            reward = self.rewards['tp']
        elif action == 0 and true_label == 0:
            reward = self.rewards['tn']
        elif action == 1 and true_label == 0:
            reward = self.rewards['fp']
        else:
            reward = self.rewards['fn']

        self.current_idx += 1
        self.steps += 1
        done = self.steps >= self.max_steps or self.current_idx >= len(self.X)
        obs = self.X[min(self.current_idx, len(self.X) - 1)]
        return obs, reward, done, False, {}


class IdsEnvSlidingWindow(gym.Env):
    """
    Experiment 2: Sliding Window Environment.
    Observes last N consecutive samples instead of a single sample.
    Hypothesis: temporal patterns → better detection of multi-step attacks.
    """
    metadata = {'render_modes': []}

    def __init__(self, data_dir=DATA_DIR, exclude_labels=None, reward_config=None,
                 window_size=5):
        super().__init__()
        self.rewards = reward_config or {'tp': 1.0, 'tn': 1.0, 'fn': -1.0, 'fp': -1.0}
        self.window_size = window_size

        X_train = pd.read_parquet(os.path.join(data_dir, 'X_train.parquet')).values.astype(np.float32)
        y_train = pd.read_parquet(os.path.join(data_dir, 'y_train.parquet')).values.ravel()
        X_test = pd.read_parquet(os.path.join(data_dir, 'X_test.parquet')).values.astype(np.float32)
        y_test = pd.read_parquet(os.path.join(data_dir, 'y_test.parquet')).values.ravel()

        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_features = X_train.shape[1]

        if exclude_labels is not None:
            raw_labels = pd.read_parquet(os.path.join(data_dir, 'y_train.parquet'))
            if 'label_raw' in raw_labels.columns:
                mask = ~raw_labels['label_raw'].isin(exclude_labels)
                self.X = self.X[mask.values]
                self.y = self.y[mask.values]

        self.action_space = gym.spaces.Discrete(2)
        obs_size = self.n_features * window_size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
        self.current_idx = 0
        self.max_steps = min(1000, len(self.X) - window_size)
        self.steps = 0
        print(f"  [Env2] Sliding window: {window_size} × {self.n_features} = {obs_size} features")

    def _get_window_obs(self):
        start = max(0, self.current_idx - self.window_size + 1)
        window = self.X[start:self.current_idx + 1]
        # Pad if at the start
        if len(window) < self.window_size:
            padding = np.zeros((self.window_size - len(window), self.n_features), dtype=np.float32)
            window = np.vstack([padding, window])
        return window.flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = np.random.randint(self.window_size, len(self.X) - self.max_steps)
        self.steps = 0
        return self._get_window_obs(), {}

    def step(self, action):
        true_label = self.y[self.current_idx]
        if action == 1 and true_label == 1:
            reward = self.rewards['tp']
        elif action == 0 and true_label == 0:
            reward = self.rewards['tn']
        elif action == 1 and true_label == 0:
            reward = self.rewards['fp']
        else:
            reward = self.rewards['fn']

        self.current_idx += 1
        self.steps += 1
        done = self.steps >= self.max_steps or self.current_idx >= len(self.X)
        obs = self._get_window_obs()
        return obs, reward, done, False, {}


class IdsEnvSequential(gym.Env):
    """
    Experiment 3: Sequential Episode Environment.
    Episodes follow dataset order (time-ordered) instead of random start.
    Hypothesis: sequential context → more realistic training dynamics.
    """
    metadata = {'render_modes': []}

    def __init__(self, data_dir=DATA_DIR, exclude_labels=None, reward_config=None):
        super().__init__()
        self.rewards = reward_config or {'tp': 1.0, 'tn': 1.0, 'fn': -1.0, 'fp': -1.0}

        self.X = pd.read_parquet(os.path.join(data_dir, 'X_train.parquet')).values.astype(np.float32)
        self.y = pd.read_parquet(os.path.join(data_dir, 'y_train.parquet')).values.ravel()
        X_test = pd.read_parquet(os.path.join(data_dir, 'X_test.parquet')).values.astype(np.float32)
        y_test = pd.read_parquet(os.path.join(data_dir, 'y_test.parquet')).values.ravel()
        self.X_test = X_test
        self.y_test = y_test

        if exclude_labels is not None:
            raw_labels = pd.read_parquet(os.path.join(data_dir, 'y_train.parquet'))
            if 'label_raw' in raw_labels.columns:
                mask = ~raw_labels['label_raw'].isin(exclude_labels)
                self.X = self.X[mask.values]
                self.y = self.y[mask.values]

        self.n_features = self.X.shape[1]
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.n_features,), dtype=np.float32)

        self.max_steps = min(1000, len(self.X))
        # Sequential: track global position
        self.global_idx = 0
        self.steps = 0
        print(f"  [Env3] Sequential episodes: {len(self.X)} samples, {self.n_features} features")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Sequential: continue from where we left off, wrap around
        if self.global_idx >= len(self.X) - self.max_steps:
            self.global_idx = 0
        self.current_idx = self.global_idx
        self.steps = 0
        return self.X[self.current_idx], {}

    def step(self, action):
        true_label = self.y[self.current_idx]
        if action == 1 and true_label == 1:
            reward = self.rewards['tp']
        elif action == 0 and true_label == 0:
            reward = self.rewards['tn']
        elif action == 1 and true_label == 0:
            reward = self.rewards['fp']
        else:
            reward = self.rewards['fn']

        self.current_idx += 1
        self.global_idx = self.current_idx
        self.steps += 1
        done = self.steps >= self.max_steps or self.current_idx >= len(self.X)
        obs = self.X[min(self.current_idx, len(self.X) - 1)]
        return obs, reward, done, False, {}
