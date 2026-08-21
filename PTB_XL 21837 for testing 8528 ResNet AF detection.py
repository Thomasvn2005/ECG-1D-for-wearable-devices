import os
import ast
import time
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from scipy.signal import butter, filtfilt, resample
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
#                     1. HARDWARE CONFIGURATION & PATHS
# ==============================================================================
PTBXL_DIR = r"D:\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CSV_PATH = os.path.join(PTBXL_DIR, "ptbxl_database.csv")
MODEL_PATH = "best_resnet50_af.pth"

TARGET_LEN = 3000  # Synchronized 10 seconds at 300 Hz
BATCH_SIZE = 128
DECISION_THRESHOLD = 0.54  # Calibrated optimal threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
if torch.cuda.is_available():
    print(
        f"[DEVICE] Hardware Acceleration: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
else:
    print("[DEVICE] Running on CPU")
print("=" * 70)


# ==============================================================================
#               2. BIOMEDICAL PREPROCESSING & SIGNAL RESAMPLING
# ==============================================================================
def butter_bandpass(lowcut=0.5, highcut=45.0, fs=300.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


B_COEFF, A_COEFF = butter_bandpass(0.5, 45.0, 300.0, order=3)


def preprocess_ptbxl_record(sig_raw, orig_fs):
    # 1. Resample to standard 300 Hz (3000 samples / 10s)
    if orig_fs != 300:
        sig_resampled = resample(sig_raw, TARGET_LEN)
    else:
        sig_resampled = sig_raw[:TARGET_LEN]

    # 2. Zero-phase 3rd-order Butterworth Bandpass Filter (0.5 - 45 Hz)
    sig_filtered = filtfilt(B_COEFF, A_COEFF, sig_resampled.astype(np.float64)).astype(np.float32)

    # 3. Robust Z-score Normalization
    mean = np.mean(sig_filtered)
    std = np.std(sig_filtered) + 1e-8
    return (sig_filtered - mean) / std


class PTBXLDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig_tensor = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return sig_tensor, label_tensor


def load_ptbxl_data(csv_path, root_dir):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Reference database file not found: {csv_path}")

    print(f"[INFO] Reading annotation file: {csv_path}")
    df = pd.read_csv(csv_path, index_col='ecg_id')
    df['scp_dict'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))

    # Binary Labeling: 'AFIB' -> 1 (Atrial Fibrillation), All others -> 0 (Non-AF)
    df['label'] = df['scp_dict'].apply(lambda d: 1 if 'AFIB' in d else 0)

    col_filename = 'filename_hr' if 'filename_hr' in df.columns else 'filename_lr'

    signals, labels = [], []
    print(f"[INFO] Parsing and extracting Lead II signals for {len(df)} records...")
    start_t = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading PTB-XL"):
        file_rel_path = row[col_filename]
        full_path = os.path.join(root_dir, file_rel_path)

        try:
            data, meta = wfdb.rdsamp(full_path)
            lead1_sig = data[:, 0]  # Extract Lead II
            proc_sig = preprocess_ptbxl_record(lead1_sig, orig_fs=meta['fs'])
            signals.append(proc_sig)
            labels.append(row['label'])
        except Exception:
            continue

    print(f"[INFO] Successfully loaded & filtered {len(signals)} records in {time.time() - start_t:.2f}s.")
    return np.array(signals), np.array(labels)


# ==============================================================================
#         3. HYBRID 1D-SE-RESNET50 + BIDIRECTIONAL GRU ARCHITECTURE
# ==============================================================================
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = self.fc(x).view(b, c, 1)
        return x * w


class SEBottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.se = SEBlock1D(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        return self.relu(out)


class HybridSEResNet50_BiGRU(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.in_planes = 64

        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.gru = nn.GRU(input_size=512 * SEBottleneck1D.expansion, hidden_size=128,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(128 * 2, num_classes)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(SEBottleneck1D(self.in_planes, planes, s))
            self.in_planes = planes * SEBottleneck1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)
        out = torch.mean(gru_out, dim=1)
        out = self.dropout(out)
        return self.fc(out)


# ==============================================================================
#                4. EXTERNAL VALIDATION INFERENCE PIPELINE
# ==============================================================================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {MODEL_PATH}")

    # 1. Load and prepare external dataset
    X_test, y_test = load_ptbxl_data(CSV_PATH, PTBXL_DIR)

    print(f"\n[EXTERNAL TEST DISTRIBUTION - PTB-XL]")
    print(f" - Total Test Samples          : {len(y_test)}")
    print(f" - Negative Class (Non-AF: 0)  : {np.sum(y_test == 0)}")
    print(f" - Positive Class (AFIB: 1)    : {np.sum(y_test == 1)}")

    test_loader = DataLoader(PTBXLDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 2. Instantiate and load pretrained weights
    print(f"\n[INFO] Loading pretrained model weights from: {MODEL_PATH}")
    model = HybridSEResNet50_BiGRU(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 3. Model Inference
    print("[INFO] Executing zero-shot inference on 21,837 samples...")
    test_probs, test_targets = [], []
    start_eval_t = time.time()

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            probs = F.softmax(model(batch_x.to(device)), dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)
            test_targets.extend(batch_y.numpy())

    print(f"[INFO] Inference completed in {time.time() - start_eval_t:.2f}s!")

    # 4. Generate Classification Report and Confusion Matrix
    test_preds = (np.array(test_probs) >= DECISION_THRESHOLD).astype(int)

    print("\n" + "#" * 70)
    print(f" EXTERNAL VALIDATION REPORT ON PTB-XL DATASET (THRESHOLD = {DECISION_THRESHOLD:.2f})")
    print("#" * 70)
    print(classification_report(test_targets, test_preds, target_names=["Non-AF (0)", "AF (1)"], digits=4))

    cm = confusion_matrix(test_targets, test_preds)
    print("Confusion Matrix:")
    print(f"                 Predicted Non-AF   Predicted AF")
    print(f"Actual Non-AF       {cm[0, 0]:<17} {cm[0, 1]}")
    print(f"Actual AF           {cm[1, 0]:<17} {cm[1, 1]}")
    print("#" * 70)


if __name__ == '__main__':
    main()