import pandas as pd
import numpy as np
import torch
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from train_rl_agent import DQN  # Import the model architecture

# Device Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_agent():
    print("Loading processed data for evaluation...")
    data_dir = '/home/abishik/HONOURS_PROJECT/processed_data'
    model_dir = '/home/abishik/HONOURS_PROJECT/models'
    
    # Load data (Updated filenames)
    X_path = os.path.join(data_dir, 'X_2017_full.parquet')
    y_path = os.path.join(data_dir, 'y_2017_binary.npy')
    
    X = pd.read_parquet(X_path).to_numpy()
    y = np.load(y_path)
    
    # Split data (Must match the baseline split for fair comparison!)
    print("Splitting data (test_size=0.2, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load the trained agent
    state_size = X.shape[1]
    action_size = 2 # Benign / Malicious
    
    print("Loading trained DQN agent...")
    qnetwork = DQN(state_size, action_size, seed=0).to(device)
    model_path = os.path.join(model_dir, 'dqn_agent.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run train_rl_agent.py first.")
        return

    qnetwork.load_state_dict(torch.load(model_path, map_location=device))
    qnetwork.eval() # Set to evaluation mode
    
    print("Running inference on Test Set...")
    predictions = []
    
    # Convert Test set to tensors
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    
    # Inference batching to avoid OOM
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size]
            output = qnetwork(batch)
            # Get action with highest Q-value
            preds = torch.argmax(output, dim=1).cpu().numpy()
            predictions.extend(preds)
            
    y_pred = np.array(predictions)
    
    # Metrics
    print("\n--- RL Agent Evaluation Results ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Compare with Baseline (Optional: Load baseline to print side-by-side if needed)
    # baseline_path = os.path.join(model_dir, 'baseline_random_forest.joblib')
    # if os.path.exists(baseline_path):
    #     rf = joblib.load(baseline_path)
    #     rf_acc = rf.score(X_test, y_test)
    #     print(f"\n(For Comparison) Baseline Random Forest Accuracy: {rf_acc:.4f}")

if __name__ == "__main__":
    evaluate_agent()
