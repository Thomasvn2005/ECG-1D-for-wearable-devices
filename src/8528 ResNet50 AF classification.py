"""
========================================================================================
PROJECT: HIGH-PERFORMANCE 4-CLASS ECG CLASSIFICATION (PHYSIONET 2017 BENCHMARK)
ARCHITECTURE: 1D-SE-ResNet50 + 2-Stage BiGRU + Dual Temporal Pooling (Mean+Max)
OPTIMIZATION: Smoothed Focal Loss (Root-Inverse Alpha) + Cosine Warmup + Temporal Cutout
CLASSES:
  - 0: Normal Rhythm (N)
  - 1: Atrial Fibrillation (A)
  - 2: Other Arrhythmia / Rhythm Abnormalities (O)
  - 3: Noisy / Uninterpretable Recording (~)
========================================================================================
"""

import os
import time
import math
import numpy as np
import pandas as pd
from scipy import io as sio
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. HARDWARE ACCELERATION & HYPERPARAMETERS
# ==============================================================================
DATA_DIR = r"D:\af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0\training2017"

TARGET_LEN = 3000                # Standardized 10 seconds @ 300 Hz
BATCH_SIZE = 128                 # Optimized for RTX Laptop GPU VRAM
EPOCHS = 50                      # Extended epochs with Warmup & Cosine Decay
WARMUP_EPOCHS = 3                # Linear warmup epochs
INITIAL_LR = 1e-3                # Peak learning rate
MIN_LR = 1e-6                    # Minimum learning rate
NUM_CLASSES = 4
MODEL_SAVE_PATH = "best_resnet50_4class_sota.pth"

LABEL_MAP = {'N': 0, 'A': 1, 'O': 2, '~': 3}
CLASS_NAMES = ['Normal (N)', 'AF (A)', 'Other (O)', 'Noisy (~)']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
if torch.cuda.is_available():
    print(f"[DEVICE] GPU Acceleration: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("[DEVICE] Running on CPU")
print("=" * 80)

# ==============================================================================
# 2. ADVANCED SIGNAL PREPROCESSING & DYNAMIC DATA AUGMENTATION
# ==============================================================================
def butter_bandpass(lowcut=0.5, highcut=45.0, fs=300.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

B_COEFF, A_COEFF = butter_bandpass(0.5, 45.0, 300.0, order=3)

def preprocess_ecg(raw_sig, target_len=3000):
    sig = np.squeeze(raw_sig).astype(np.float64)
    # Zero-phase Butterworth Bandpass Filter (0.5 - 45.0 Hz)
    if len(sig) > 15:
        sig = filtfilt(B_COEFF, A_COEFF, sig)
    sig = sig.astype(np.float32)

    # Pad or Truncate to exact target length
    if len(sig) >= target_len:
        sig = sig[:target_len]
    else:
        sig = np.pad(sig, (0, target_len - len(sig)), mode='constant')

    # Robust Z-Score Normalization
    mean = np.mean(sig)
    std = np.std(sig) + 1e-8
    return (sig - mean) / std

def augment_ecg(sig):
    """Multi-stage time-series augmentation for ECG robust learning."""
    # 1. Circular temporal shift (+-250 samples)
    shift = np.random.randint(-250, 250)
    sig = np.roll(sig, shift)

    # 2. Dynamic Amplitude Scaling
    if np.random.rand() > 0.5:
        scale = np.random.uniform(0.85, 1.15)
        sig = sig * scale

    # 3. Additive Gaussian White Noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.015, size=sig.shape).astype(np.float32)
        sig = sig + noise

    # 4. Temporal Cutout (Random Zero-Masking 100-200 samples)
    if np.random.rand() > 0.6:
        mask_len = np.random.randint(100, 200)
        start_idx = np.random.randint(0, len(sig) - mask_len)
        sig[start_idx:start_idx + mask_len] = 0.0

    return sig.astype(np.float32)

class PhysioNet4ClassDataset(Dataset):
    def __init__(self, signals, labels, is_train=False):
        self.signals = signals
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig = self.signals[idx].copy()
        if self.is_train:
            sig = augment_ecg(sig)

        sig_tensor = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 3000)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return sig_tensor, label_tensor

def load_multiclass_data(data_dir):
    possible_names = ["REFERENCE.csv", "REFERENCE", "REFERENCE-v3.csv", "REFERENCE-v3"]
    ref_file = None
    for name in possible_names:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            ref_file = p
            break

    if ref_file is None:
        raise FileNotFoundError(f"Reference label file not found in: {data_dir}")

    print(f"[INFO] Loading reference annotation file: {ref_file}")
    df = pd.read_csv(ref_file, header=None, names=['record_name', 'label'])

    df['clean_label'] = df['label'].astype(str).str.strip()
    df = df[df['clean_label'].isin(LABEL_MAP.keys())].copy()
    df['multiclass_label'] = df['clean_label'].map(LABEL_MAP)

    signals, labels = [], []
    print("[INFO] Filtering and loading all single-lead ECG recordings into RAM...")
    start_t = time.time()

    for _, row in df.iterrows():
        record = str(row['record_name']).strip()
        mat_path = os.path.join(data_dir, f"{record}.mat")
        if not os.path.exists(mat_path):
            mat_path = os.path.join(data_dir, record)

        if os.path.exists(mat_path):
            mat_data = sio.loadmat(mat_path)
            raw_sig = mat_data['val'][0]
            signals.append(preprocess_ecg(raw_sig, TARGET_LEN))
            labels.append(row['multiclass_label'])

    print(f"[INFO] Successfully loaded & filtered {len(signals)} records in {time.time() - start_t:.2f}s.")
    return np.array(signals), np.array(labels)

# ==============================================================================
# 3. 1D-SE-RESNET50 + 2-STAGE BiGRU + DUAL TEMPORAL POOLING
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

class HybridSEResNet50_DualPool_BiGRU(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.in_planes = 64

        # Stem Layer
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 4 Residual Stages
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # 2-Layer Bidirectional GRU
        self.gru = nn.GRU(
            input_size=512 * SEBottleneck1D.expansion,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # High-Capacity Classification Head with LayerNorm
        # Dual Pooling (Mean + Max) output dimension: (128 * 2) * 2 = 512
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(SEBottleneck1D(self.in_planes, planes, s))
            self.in_planes = planes * SEBottleneck1D.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))  # Shape: (B, 2048, L)
        x = x.permute(0, 2, 1)                                     # Shape: (B, L, 2048)
        gru_out, _ = self.gru(x)                                   # Shape: (B, L, 256)

        # Dual Temporal Pooling: Mean + Max captures global rhythm and isolated ectopic spikes
        mean_pool = torch.mean(gru_out, dim=1)
        max_pool, _ = torch.max(gru_out, dim=1)
        dual_pooled = torch.cat([mean_pool, max_pool], dim=1)      # Shape: (B, 512)

        return self.head(dual_pooled)

# ==============================================================================
# 4. SMOOTHED MULTI-CLASS FOCAL LOSS WITH LABEL REGULARIZATION
# ==============================================================================
class SmoothedMultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

# ==============================================================================
# 5. TRAINING LOOP WITH WARMUP & PHYSIONET CHALLENGE BENCHMARK
# ==============================================================================
def compute_challenge_f1(targets, preds):
    f1_scores = f1_score(targets, preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
    f1_challenge = np.mean(f1_scores[:3])
    return f1_challenge, f1_scores

def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    X, y = load_multiclass_data(DATA_DIR)

    # Stratified Train/Val/Test Split (70% / 10% / 20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
    )

    print(f"\n[CLASS DISTRIBUTION]")
    for i, name in enumerate(CLASS_NAMES):
        print(f" - {name:<12}: Train = {np.sum(y_train==i):<5} | Val = {np.sum(y_val==i):<4} | Test = {np.sum(y_test==i)}")

    # Balanced Root-Inverse Alpha Weights (Smoothed)
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    alpha_weights = 1.0 / np.sqrt(class_counts + 1e-6)
    alpha_weights = alpha_weights / np.sum(alpha_weights) * NUM_CLASSES
    alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(PhysioNet4ClassDataset(X_train, y_train, is_train=True), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(PhysioNet4ClassDataset(X_val, y_val, is_train=False), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(PhysioNet4ClassDataset(X_test, y_test, is_train=False), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = HybridSEResNet50_DualPool_BiGRU(num_classes=NUM_CLASSES).to(device)
    criterion = SmoothedMultiClassFocalLoss(alpha=alpha_tensor, gamma=1.5, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-3)
    scheduler = get_lr_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS, INITIAL_LR, MIN_LR)

    best_val_challenge_f1 = 0.0

    print("\n" + "=" * 80)
    print(" STARTING SOTA 4-CLASS TRAINING LOOP (SE-RESNET50 + DUAL-POOL BiGRU)")
    print("=" * 80)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping to ensure optimization stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x.to(device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())

        val_challenge_f1, per_class_f1 = compute_challenge_f1(np.array(val_targets), np.array(val_preds))
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Loss: {avg_train_loss:.4f} | "
              f"Val Challenge F1: {val_challenge_f1:.4f} (N:{per_class_f1[0]:.3f}, A:{per_class_f1[1]:.3f}, O:{per_class_f1[2]:.3f}, ~:{per_class_f1[3]:.3f}) | LR: {current_lr:.6f}")

        scheduler.step()

        if val_challenge_f1 > best_val_challenge_f1:
            best_val_challenge_f1 = val_challenge_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n" + "=" * 80)
    print(f" TRAINING COMPLETE! Best Validation Challenge F1 = {best_val_challenge_f1:.4f}")
    print(f" Model saved to: {MODEL_SAVE_PATH}")
    print("=" * 80)

    # 6. BENCHMARK EVALUATION ON UNSEEN TEST SET
    print("\n[INFO] Loading best model weights for holdout test evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(batch_y.numpy())

    test_challenge_f1, _ = compute_challenge_f1(np.array(test_targets), np.array(test_preds))

    print("\n" + "#" * 80)
    print(f" FINAL SOTA BENCHMARK ON TEST SET (CHALLENGE F1 = {test_challenge_f1:.4f})")
    print("#" * 80)
    print(classification_report(test_targets, test_preds, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(test_targets, test_preds)
    print("Confusion Matrix:")
    print(f"{'':<16} Predicted N   Predicted A   Predicted O   Predicted ~")
    for i, name in enumerate(CLASS_NAMES):
        print(f"Actual {name:<10} {cm[i, 0]:<13} {cm[i, 1]:<13} {cm[i, 2]:<13} {cm[i, 3]}")
    print("#" * 80)

if __name__ == '__main__':
    main()
