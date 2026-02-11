# Data Preprocessing for Multi-Scenario Experiments
# This script creates properly separated datasets for ML (train/test split) and RL (full dataset replay)
# Updated to handle CIC-IDS2017 and CIC-IoT-2023 with Feature Mapping

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Paths
Data_Dir_2017 = '/home/abishik/HONOURS_PROJECT/data'
Data_Dir_2023 = '/home/abishik/HONOURS_PROJECT/data/CIC-IoT-2023'
Processed_Dir = '/home/abishik/HONOURS_PROJECT/processed_data'
os.makedirs(Processed_Dir, exist_ok=True)

# Feature Mapping for Generalization (IoT 2023 -> IDS 2017 Names)
# We only map features that exist in BOTH datasets
IOT_TO_IDS_MAPPING = {
    'Protocol Type': 'protocol',
    'fin_flag_number': 'fin_flag_count',
    'syn_flag_number': 'syn_flag_count',
    'rst_flag_number': 'rst_flag_count',
    'psh_flag_number': 'psh_flag_count',
    'ack_flag_number': 'ack_flag_count',
    'ece_flag_number': 'ece_flag_count', # cwe/ece might differ, but let's try
    'AVG': 'packet_length_mean',
    'Std': 'packet_length_std',
    'Variance': 'packet_length_variance',
    'IAT': 'flow_iat_mean', 
    'Tot size': 'total_length_of_fwd_packets', # Approx? Or Total Length? Let's use as proxy
}

COMMON_FEATURES = list(IOT_TO_IDS_MAPPING.values())

def load_cic_ids2017(data_dir):
    """Load and clean CIC-IDS2017 data."""
    print(f"\n[CIC-IDS2017] Loading from {data_dir}...")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    # Exclude the IoT folder if it was caught by glob (it shouldn't be if looking for *.csv files in root, but careful)
    csv_files = [f for f in csv_files if "CIC-IoT-2023" not in f]
    
    if not csv_files:
        print("waring: No 2017 files found in root data dir!")
        return None

    df_list = []
    for file_path in csv_files:
        print(f"  Loading: {os.path.basename(file_path)}")
        try:
            # Skip bad lines, large chunks
            chunks = pd.read_csv(file_path, encoding='latin1', chunksize=200000, on_bad_lines='skip')
            for chunk in chunks:
                df_list.append(chunk)
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
    
    if not df_list:
        return None

    df = pd.concat(df_list, ignore_index=True)
    
    # Standardize names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
    
    # Basic Cleaning
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Label cleaning
    df.rename(columns={'label': 'Label'}, inplace=True) # Normalize label col name temporarily
    df['Label'] = df['Label'].astype(str).str.strip().str.lower()
    
    # Clean binary label
    df['label_binary'] = (df['Label'] != 'benign').astype(int)
    
    print(f"  2017 Data Shape: {df.shape}")
    return df

def load_cic_iot2023(data_dir):
    """Load and clean CIC-IoT-2023 data."""
    print(f"\n[CIC-IoT-2023] Loading from {data_dir}...")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print("  Warning: No 2023 files found!")
        return None

    df_list = []
    for file_path in csv_files:
        print(f"  Loading: {os.path.basename(file_path)}")
        try:
            chunk = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip') # Files are smaller, read full?
            # Add a 'Label' column based on filename if missing, but usually distinct files have specific attacks
            # Check filename for label
            fname = os.path.basename(file_path).lower()
            if 'benign' in fname:
                chunk['Label'] = 'benign'
            elif 'ddos' in fname:
                chunk['Label'] = 'ddos'
            elif 'sql' in fname:
                chunk['Label'] = 'web_attack'
            elif 'xss' in fname:
                chunk['Label'] = 'web_attack'
            elif 'mirai' in fname:
                chunk['Label'] = 'botnet'
            else:
                chunk['Label'] = 'unknown_attack'
            
            df_list.append(chunk)
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")

    if not df_list:
        return None
        
    df = pd.concat(df_list, ignore_index=True)
    
    # Clean NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print(f"  2023 Data Shape: {df.shape}")
    return df

def preprocess_standard_2017(df, output_dir):
    """Process full 2017 dataset for Standard & Zero-Day Scenarios (High Feature Count)."""
    print("\n[Processing] Standard CIC-IDS2017 (Full Features)")
    
    # Drop mapped columns to avoid confusion? No, keep all.
    # Drop irrelevant
    cols_to_drop = ['flow_id', 'source_ip', 'destination_ip', 'timestamp', 'label', 'Label'] # keep label_binary
    X = df.drop([c for c in cols_to_drop if c in df.columns], axis=1, errors='ignore')
    X = X.drop('label_binary', axis=1, errors='ignore') # Separate X
    
    y = df['label_binary']
    y_labels = df['Label'] # String labels for stratifying zero-day
    
    # Scale
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Save Full for RL
    X_scaled.to_parquet(os.path.join(output_dir, 'X_2017_full.parquet'), index=False)
    np.save(os.path.join(output_dir, 'y_2017_binary.npy'), y.values)
    
    # Save Label Strings for Zero-Day Logic (Critical!)
    # We need to filter by 'DDoS' or 'Web Attack' later
    y_labels.to_csv(os.path.join(output_dir, 'y_2017_labels.csv'), index=False)
    
    # Train/Test Split for ML Baselines
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save ML Sets
    X_train.to_parquet(os.path.join(output_dir, 'X_train_2017.parquet'), index=False)
    X_test.to_parquet(os.path.join(output_dir, 'X_test_2017.parquet'), index=False)
    np.save(os.path.join(output_dir, 'y_train_2017.npy'), y_train.values)
    np.save(os.path.join(output_dir, 'y_test_2017.npy'), y_test.values)
    
    joblib.dump(scaler, os.path.join(output_dir, 'scaler_2017_full.joblib'))
    print("  Saved Standard 2017 Datasets.")

def preprocess_generalization(df_2017, df_2023, output_dir):
    """Process both datasets using ONLY Common Features for Generalization Test."""
    print("\n[Processing] Generalization Sets (Common Features Only)")
    
    if df_2023 is None:
        print("  Skipping Generalization (No 2023 data)")
        return

    # 1. Map 2023 columns to 2017 names
    df_2023_mapped = df_2023.rename(columns=IOT_TO_IDS_MAPPING)
    
    # 2. Select ONLY common features
    features = COMMON_FEATURES
    
    # Verify features exist in both
    missing_2017 = [f for f in features if f not in df_2017.columns]
    missing_2023 = [f for f in features if f not in df_2023_mapped.columns]
    
    if missing_2017 or missing_2023:
        print(f"  Error: Missing features for mapping.\n  2017 missing: {missing_2017}\n  2023 missing: {missing_2023}")
        return

    X_2017 = df_2017[features]
    X_2023 = df_2023_mapped[features]
    
    y_2017 = df_2017['label_binary']
    # Create binary label for 2023
    y_2023 = (df_2023_mapped['Label'] != 'benign').astype(int)
    
    # 3. Train Scaler on 2017 ONLY (Assumption: Training data defines the scale)
    scaler_gen = MinMaxScaler()
    scaler_gen.fit(X_2017)
    
    X_2017_scaled = pd.DataFrame(scaler_gen.transform(X_2017), columns=features)
    X_2023_scaled = pd.DataFrame(scaler_gen.transform(X_2023), columns=features)
    
    # 4. Save
    X_2017_scaled.to_parquet(os.path.join(output_dir, 'X_gen_train_2017.parquet'), index=False)
    np.save(os.path.join(output_dir, 'y_gen_train_2017.npy'), y_2017.values)
    
    X_2023_scaled.to_parquet(os.path.join(output_dir, 'X_gen_test_2023.parquet'), index=False)
    np.save(os.path.join(output_dir, 'y_gen_test_2023.npy'), y_2023.values)
    
    print("  Saved Generalization Datasets (Train on 2017, Test on 2023).")

if __name__ == "__main__":
    print("="*50)
    print("Starting Data Preprocessing Pipeline")
    print("="*50)
    
    # 1. Load Data
    df_2017 = load_cic_ids2017(Data_Dir_2017)
    df_2023 = load_cic_iot2023(Data_Dir_2023)
    
    if df_2017 is not None:
        # 2. Process Standard (Full 2017)
        preprocess_standard_2017(df_2017, Processed_Dir)
        
        # 3. Process Generalization (Common Only)
        if df_2023 is not None:
            preprocess_generalization(df_2017, df_2023, Processed_Dir)
            
    print("\nProcessing Complete.")
