# Fast Batch Evaluation for PPO Agent
import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_ppo_batch():
    data_dir = '/home/abishik/HONOURS_PROJECT/processed_data'
    model_dir = '/home/abishik/HONOURS_PROJECT/models'
    
    print("Loading test data...")
    X_test = pd.read_parquet(os.path.join(data_dir, 'X_test_2017.parquet')).to_numpy()
    y_test = np.load(os.path.join(data_dir, 'y_test_2017.npy'))
    
    print(f"Test samples: {len(X_test)}")
    
    print("Loading PPO model...")
    model = PPO.load(os.path.join(model_dir, 'ppo_agent'))
    
    print("Running batch inference...")
    predictions = []
    batch_size = 10000
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size].astype(np.float32)
        actions, _ = model.predict(batch, deterministic=True)
        predictions.extend(actions)
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(X_test))}/{len(X_test)}")
    
    y_pred = np.array(predictions)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print("PPO AGENT EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    print(f"{'='*50}")

if __name__ == "__main__":
    evaluate_ppo_batch()
