# ECG AF/AFL Detection Using 1D ViT-Small
The structure for now includes cell 1 and cell 2
## Cell 1: Data Loading and Model Architecture
- Loads four preprocessed ECG datasets stored in HDF5: PTB-XL, LTAFDB , SHDB-AF, and MIT-AFDB.
- Prepares single-lead ECG signals sampled at 125 Hz with 10-second windows (1,250 samples).
- Implements 1D ViT-Small with:
  - 12 Transformer blocks
  - 6 attention heads
  - 384 embedding dimensions
  - 25 patches (50 samples per patch)
  - Learnable positional embeddings and CLS token
- Creates PyTorch DataLoaders.
- Verifies dataset integrity, input shapes, and model forward propagation.
## Cell 2: Model Training and Validation
- Trains ViT-Small using PTB-XL and LTAFDB V3[1].
- Performs binary classification:
  - Class 0: Non-AF/AFL
  - Class 1: AF/AFL
- Training configuration:
  - Optimizer: AdamW
  - Learning rate: 0.0003
  - Weight decay: 0.0001
  - Loss function: CrossEntropyLoss
  - Batch size: 128
  - Epochs: 20
- Applies ECG data augmentation.
- Validates using SHDB-AF.
- Evaluates Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Supports automatic checkpoint saving and training resumption.
- Saves the latest, best, and final model checkpoints to Google Drive.
## Dataset Configuration
- Training Dataset: PTB-XL, LTAFDB
- Validating Dataset: SHDBAF
- Testing Dataset: MIT-AFDB


