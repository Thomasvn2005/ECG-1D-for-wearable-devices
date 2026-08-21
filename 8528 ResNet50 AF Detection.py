import os
import time
import numpy as np
import pandas as pd
from scipy import io as sio
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
#                   1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
DATA_DIR = r"D:\af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0\training2017"

TARGET_LEN = 3000  # 10 seconds at 300 Hz sampling rate
BATCH_SIZE = 128
EPOCHS = 45  # Sufficient epochs for CNN + BiGRU convergence
INITIAL_LR = 1e-3  # Initial learning rate
MODEL_SAVE_PATH = "best_resnet50_af.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
if torch.cuda.is_available():
    print(
        f"[DEVICE] GPU: {torch.cuda.get_device_name(0)} | Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
else:
    print("[DEVICE] Running on CPU")
print("=" * 70)


# ==============================================================================
#           2. BIOMEDICAL SIGNAL PROCESSING & DATA AUGMENTATION
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
    # Zero-phase Butterworth Bandpass Filter
    if len(sig) > 15:
        sig = filtfilt(B_COEFF, A_COEFF, sig)
    sig = sig.astype(np.float32)

    # Pad or Truncate to exact length
    if len(sig) >= target_len:
        sig = sig[:target_len]
    else:
        sig = np.pad(sig, (0, target_len - len(sig)), mode='constant')

    # Robust Z-score Normalization
    mean = np.mean(sig)
    std = np.std(sig) + 1e-8
    return (sig - mean) / std


def augment_ecg(sig):
    """Dynamic ECG Data Augmentation on Training Set."""
    # 1. Random Time Shift (+- 200 samples)
    shift = np.random.randint(-200, 200)
    sig = np.roll(sig, shift)

    # 2. Random Amplitude Scaling
    if np.random.rand() > 0.5:
        scale = np.random.uniform(0.85, 1.15)
        sig = sig * scale

    # 3. Additive Gaussian White Noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.02, size=sig.shape).astype(np.float32)
        sig = sig + noise

    return sig.astype(np.float32)


class PhysioNetDataset(Dataset):
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


def load_data(data_dir):
    possible_names = ["REFERENCE.csv", "REFERENCE", "REFERENCE-v3.csv", "REFERENCE-v3"]
    ref_file = None
    for name in possible_names:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            ref_file = p
            break

    if ref_file is None:
        raise FileNotFoundError(f"Reference label file not found in: {data_dir}")

    print(f"[INFO] Loading reference label file: {ref_file}")
    df = pd.read_csv(ref_file, header=None, names=['record_name', 'label'])
    df['binary_label'] = df['label'].apply(lambda x: 1 if str(x).strip() == 'A' else 0)

    signals, labels = [], []
    print("[INFO] Filtering and loading all ECG records into RAM...")
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
            labels.append(row['binary_label'])

    print(f"[INFO] Successfully loaded & filtered {len(signals)} records in {time.time() - start_t:.2f}s.")
    return np.array(signals), np.array(labels)


# ==============================================================================
#       3. HYBRID ARCHITECTURE: 1D-SE-RESNET50 + BIDIRECTIONAL GRU
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

        # BiGRU Layer to model long-range rhythm irregularities
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
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))  # Shape: (B, 2048, L)

        # Reshape for Temporal Sequence Modeling: (Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)

        # Temporal Average Pooling across sequence steps
        out = torch.mean(gru_out, dim=1)
        out = self.dropout(out)
        return self.fc(out)


# ==============================================================================
#                          4. FOCAL LOSS OBJECTIVE
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ==============================================================================
#              5. TRAINING LOOP & DYNAMIC THRESHOLD CALIBRATION
# ==============================================================================
def main():
    X, y = load_data(DATA_DIR)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
    )

    print(f"\n[DATASET DISTRIBUTION]")
    print(f" - Train set      : {len(X_train)} samples (Non-AF: {np.sum(y_train == 0)}, AF: {np.sum(y_train == 1)})")
    print(f" - Validation set : {len(X_val)} samples (Non-AF: {np.sum(y_val == 0)}, AF: {np.sum(y_val == 1)})")
    print(f" - Internal Test  : {len(X_test)} samples (Non-AF: {np.sum(y_test == 0)}, AF: {np.sum(y_test == 1)})")

    train_loader = DataLoader(PhysioNetDataset(X_train, y_train, is_train=True), batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(PhysioNetDataset(X_val, y_val, is_train=False), batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True)
    test_loader = DataLoader(PhysioNetDataset(X_test, y_test, is_train=False), batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True)

    model = HybridSEResNet50_BiGRU(num_classes=2).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_f1 = 0.0

    print("\n" + "=" * 70)
    print(" STARTING TRAINING LOOP (SE-RESNET50 + BiGRU + FOCAL LOSS)")
    print("=" * 70)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                probs = F.softmax(model(batch_x), dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(batch_y.numpy())

        # Evaluate at default threshold 0.5
        val_preds_default = (np.array(val_probs) >= 0.5).astype(int)
        val_f1 = f1_score(val_targets, val_preds_default, pos_label=1, zero_division=0)
        val_prec = precision_score(val_targets, val_preds_default, pos_label=1, zero_division=0)
        val_rec = recall_score(val_targets, val_preds_default, pos_label=1, zero_division=0)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | "
              f"Val AF -> F1: {val_f1:.4f} (Prec: {val_prec:.3f}, Rec: {val_rec:.3f}) | LR: {current_lr:.6f}")

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n" + "=" * 70)
    print(f" TRAINING COMPLETE! Best Val F1 = {best_val_f1:.4f}")
    print(f" Model weights saved to: {MODEL_SAVE_PATH}")
    print("=" * 70)

    # 6. TEST SET EVALUATION WITH OPTIMAL THRESHOLD CALIBRATION
    print("\n[INFO] Loading best checkpoint for threshold calibration & evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Collect Validation Probabilities for Decision Boundary Tuning
    val_probs, val_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            probs = F.softmax(model(batch_x.to(device)), dim=1)[:, 1].cpu().numpy()
            val_probs.extend(probs)
            val_targets.extend(batch_y.numpy())

    best_thresh = 0.5
    best_thresh_f1 = 0.0
    for t in np.arange(0.20, 0.70, 0.02):
        score = f1_score(val_targets, (np.array(val_probs) >= t).astype(int), pos_label=1, zero_division=0)
        if score > best_thresh_f1:
            best_thresh_f1 = score
            best_thresh = t

    print(
        f"[INFO] Optimal Decision Threshold from Validation: {best_thresh:.2f} (Calibrated Val F1: {best_thresh_f1:.4f})")

    # Evaluate on Unseen Test Set
    test_probs, test_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            probs = F.softmax(model(batch_x.to(device)), dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)
            test_targets.extend(batch_y.numpy())

    test_preds_opt = (np.array(test_probs) >= best_thresh).astype(int)

    print("\n" + "#" * 70)
    print(f" FINAL SOTA EVALUATION ON INTERNAL TEST SET (THRESHOLD = {best_thresh:.2f})")
    print("#" * 70)
    print(classification_report(test_targets, test_preds_opt, target_names=["Non-AF (0)", "AF (1)"], digits=4))

    cm = confusion_matrix(test_targets, test_preds_opt)
    print("Confusion Matrix:")
    print(f"                 Predicted Non-AF   Predicted AF")
    print(f"Actual Non-AF       {cm[0, 0]:<17} {cm[0, 1]}")
    print(f"Actual AF           {cm[1, 0]:<17} {cm[1, 1]}")
    print("#" * 70)


if __name__ == '__main__':
    main()