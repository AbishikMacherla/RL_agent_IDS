# IDS Environment for RL Agent Training
# Supports zero-day simulation via label exclusion

import gymnasium as gym
import numpy as np
import pandas as pd
import os

class IdsEnv(gym.Env):
    """
    Custom Gymnasium Environment for Intrusion Detection System.
    
    The agent observes network traffic features and decides:
        Action 0 = Allow (Benign prediction)
        Action 1 = Block (Malicious prediction)
    
    Args:
        data_dir: Path to processed data files.
        exclude_labels: List of raw label strings to exclude (for zero-day testing).
                        Example: ['ddos', 'dos hulk'] excludes these from training.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir='/home/abishik/HONOURS_PROJECT/processed_data', exclude_labels=None, reward_config=None):
        super(IdsEnv, self).__init__()
        
        # Configurable reward structure (defaults to security-first 10:1)
        self.rewards = reward_config or {'tp': 10.0, 'tn': 1.0, 'fn': -10.0, 'fp': -1.0}
        
        # Load the full RL dataset (New 2017 Standard Naming)
        X_path = os.path.join(data_dir, 'X_2017_full.parquet')
        y_binary_path = os.path.join(data_dir, 'y_2017_binary.npy')
        y_raw_path = os.path.join(data_dir, 'y_2017_labels.csv')
        
        # Fallback to old file names if new ones don't exist
        if not os.path.exists(X_path):
            X_path = os.path.join(data_dir, 'X_processed.parquet')
            y_binary_path = os.path.join(data_dir, 'y_encoded.npy')
            y_raw_path = None
        
        self.X = pd.read_parquet(X_path).to_numpy()
        self.y = np.load(y_binary_path)
        
        # For zero-day exclusion (Load CSV)
        if y_raw_path and os.path.exists(y_raw_path):
             self.raw_labels = pd.read_csv(y_raw_path).iloc[:, 0].values
        else:
             self.raw_labels = None
        
        # Apply label exclusion if specified
        if exclude_labels and self.raw_labels is not None:
            mask = ~np.isin(self.raw_labels, exclude_labels)
            self.X = self.X[mask]
            self.y = self.y[mask]
            print(f"[IdsEnv] Excluded labels: {exclude_labels}")
            print(f"[IdsEnv] Remaining samples: {len(self.X)}")
        
        self.current_step = 0
        self.max_steps = len(self.X) - 1
        
        # Action space: Allow (0) or Block (1)
        self.action_space = gym.spaces.Discrete(2)
        
        # Observation space: scaled features [0, 1]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.X.shape[1],), dtype=np.float32
        )
        
        print(f"[IdsEnv] Initialized with {len(self.X)} samples, {self.X.shape[1]} features")

    def reset(self, seed=None, options=None):
        """Reset environment to a random starting point."""
        super().reset(seed=seed)
        
        # IMPORTANT: Random start to prevent overfitting to initial rows
        max_start = max(0, len(self.X) - 2000)
        self.current_step = np.random.randint(0, max_start) if max_start > 0 else 0
        
        observation = self.X[self.current_step].astype(np.float32)
        return observation, {}

    def step(self, action):
        """Execute one step: agent classifies current traffic."""
        true_label = self.y[self.current_step]
        
        # Reward based on configurable reward structure
        if action == true_label:
            reward = self.rewards['tp'] if action == 1 else self.rewards['tn']
        else:
            reward = self.rewards['fn'] if true_label == 1 else self.rewards['fp']
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        if not terminated:
            observation = self.X[self.current_step].astype(np.float32)
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        return observation, reward, terminated, False, {}

    def render(self, mode='human'):
        """Render not needed for this environment."""
        pass
