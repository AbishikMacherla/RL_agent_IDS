# Dataset Download Instructions

This guide explains how to download the datasets needed for the RL-Enhanced IDS project.

## 1. CIC-IDS2017 (Training Dataset)

**Already downloaded**: Check if you have files in `/home/abishik/HONOURS_PROJECT/data/`

If not, follow these steps:

### Option A: Direct Download (Recommended)
1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Scroll down to "Dataset Download"
3. Download the CSV files (MachineLearningCVE folder)
4. Look for files like:
   - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   - etc.
5. Place all CSV files in: `/home/abishik/HONOURS_PROJECT/data/`

### Option B: Kaggle
1. Visit: https://www.kaggle.com/datasets/cicdataset/cicids2017
2. Click "Download" (requires Kaggle account)
3. Extract and place CSVs in: `/home/abishik/HONOURS_PROJECT/data/`

---

## 2. CIC-IoT-2023 (Generalization Testing - Optional)

This is for testing trained models on unseen data. Download only if time permits.

### Step-by-Step:
1. Visit: https://www.unb.ca/cic/datasets/iotdataset-2023.html or Kaggle
2. You do **NOT** need to download everything (it's huge).
3. Download the **first CSV file** from inside these specific folders (e.g., just `Benign_pcap.csv`, you don't need `Benign_pcap1.csv` etc.):
   - **`Benign_Final/`** (Required: Normal traffic)
   - **`DDoS-TCP_Flood/`** (For DDoS scenario testing)
   - **`DDoS-UDP_Flood/`** (For DDoS scenario testing)
   - **`SqlInjection/`** (For Web Attack scenario)
   - **`XSS/`** (For Web Attack scenario)
   - **`Mirai-greeth_flood/`** (Optional: Good for showing detection of IoT-specific botnets)

4. Create folder: `/home/abishik/HONOURS_PROJECT/data/CIC-IoT-2023/`
5. Place the downloaded **CSV** files inside there.

---

## Verify Your Setup

After downloading, your folder structure should look like:

```
/home/abishik/HONOURS_PROJECT/
├── data/
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv  (CIC-IDS2017)
│   ├── [other CIC-IDS2017 CSV files...]
│   └── CIC-IoT-2023/                                     (Optional)
│       └── [CIC-IoT-2023 CSV files...]
```

---

## Run Preprocessing

Once data is in place:

```bash
cd /home/abishik/HONOURS_PROJECT
python data_preprocessing.py
```

This will create processed files in `/home/abishik/HONOURS_PROJECT/processed_data/`

---

## Notes

- CIC-IDS2017 is approximately 3GB total
- CIC-IoT-2023 is larger (several GB) - only download if needed
- The preprocessing script handles both datasets automatically
