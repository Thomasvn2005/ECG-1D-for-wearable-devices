import os,h5py,numpy as np,torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
# ============================================================
# 1. CONFIG
# ============================================================
TRAIN_H5="/content/drive/MyDrive/AF_ConvNeXtV2/preprocessed/PTBxl-processed/500hz processed/ptbxl_125hz_10s_leadI.h5"
VAL_H5="/content/drive/MyDrive/AF_ConvNeXtV2/preprocessed/SHDB-AF/shdb_af_125hz_10s_ECG1_labeled.h5"
TEST_H5="/content/drive/MyDrive/AF_ConvNeXtV2/preprocessed/MIT-AFDB/mit_afdb_125hz_10s_ECG1_test.h5"
BATCH_SIZE=128
NUM_WORKERS=0
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",DEVICE)
if DEVICE.type=="cuda":
    print("GPU:",torch.cuda.get_device_name(0))
# ============================================================
# 2. HDF5 DATASET - LAZY LOADING
# ============================================================
class ECGH5Dataset(Dataset):
    def __init__(self,h5_path,valid_labels_only=False):
        self.h5_path=str(h5_path)
        self.valid_labels_only=valid_labels_only
        self.h5=None
        self.x=None
        self.y=None
        with h5py.File(self.h5_path,"r") as f:
            n=len(f["y"])
            if valid_labels_only:
                labels=np.asarray(f["y"][:])
                self.indices=np.where((labels==0)|(labels==1))[0].astype(np.int64)
            else:
                self.indices=np.arange(n,dtype=np.int64)
    def _open(self):
        if self.h5 is None:
            self.h5=h5py.File(self.h5_path,"r")
            self.x=self.h5["x"]
            self.y=self.h5["y"]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,idx):
        self._open()
        real_idx=int(self.indices[idx])
        x=np.asarray(self.x[real_idx],dtype=np.float32)
        y=int(self.y[real_idx])
        x=torch.from_numpy(x)
        y=torch.tensor(y,dtype=torch.long)
        return x,y
    def __getstate__(self):
        state=self.__dict__.copy()
        state["h5"]=None
        state["x"]=None
        state["y"]=None
        return state
    def __del__(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except:
            pass
# ============================================================
# 3. CREATE DATASETS
# PTB-XL = TRAIN
# SHDB-AF = VALIDATION, y=-1 removed
# MIT-AFDB = TEST
# ============================================================
train_dataset=ECGH5Dataset(TRAIN_H5,valid_labels_only=False)
val_dataset=ECGH5Dataset(VAL_H5,valid_labels_only=True)
test_dataset=ECGH5Dataset(TEST_H5,valid_labels_only=False)
print("\n========== DATASET SIZE ==========")
print("Train PTB-XL:",len(train_dataset))
print("Validation SHDB-AF:",len(val_dataset))
print("Test MIT-AFDB:",len(test_dataset))
# ============================================================
# 4. DATALOADERS
# ============================================================
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=False,drop_last=False)
val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=False,drop_last=False)
test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=False,drop_last=False)
# ============================================================
# 5. CHECK ONE BATCH
# ============================================================
xb,yb=next(iter(train_loader))
print("\n========== TRAIN BATCH CHECK ==========")
print("x batch shape:",xb.shape)
print("y batch shape:",yb.shape)
print("x dtype:",xb.dtype)
print("y dtype:",yb.dtype)
print("labels in batch:",torch.unique(yb))
# ============================================================
# 6. LAYERNORM FOR (B,C,L)
# ============================================================
class LayerNorm1D(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.norm=nn.LayerNorm(channels)
    def forward(self,x):
        x=x.transpose(1,2)
        x=self.norm(x)
        x=x.transpose(1,2)
        return x
# ============================================================
# 7. GLOBAL RESPONSE NORMALIZATION - CONVNEXT V2
# Input shape here: (B,L,C)
# ============================================================
class GRN(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.gamma=nn.Parameter(torch.zeros(1,1,dim))
        self.beta=nn.Parameter(torch.zeros(1,1,dim))
    def forward(self,x):
        gx=torch.norm(x,p=2,dim=1,keepdim=True)
        nx=gx/(gx.mean(dim=-1,keepdim=True)+1e-6)
        return x+self.gamma*(x*nx)+self.beta
# ============================================================
# 8. CONVNEXT V2 BLOCK
# Paper:
# Depthwise Conv1D kernel=7
# LayerNorm
# Pointwise expansion x4
# GELU
# GRN
# Pointwise reduction
# Residual connection
# ============================================================
class ConvNeXtV2Block1D(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dwconv=nn.Conv1d(dim,dim,kernel_size=7,padding=3,groups=dim)
        self.norm=nn.LayerNorm(dim)
        self.pwconv1=nn.Linear(dim,4*dim)
        self.act=nn.GELU()
        self.grn=GRN(4*dim)
        self.pwconv2=nn.Linear(4*dim,dim)
    def forward(self,x):
        residual=x
        x=self.dwconv(x)
        x=x.transpose(1,2)
        x=self.norm(x)
        x=self.pwconv1(x)
        x=self.act(x)
        x=self.grn(x)
        x=self.pwconv2(x)
        x=x.transpose(1,2)
        x=x+residual
        return x
# ============================================================
# 9. 1D CONVNEXT V2
# Figure 2:
# depths = [3,3,9,3]
# dims   = [16,32,64,128]
# ============================================================
class ConvNeXtV2_1D(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        dims=[16,32,64,128]
        depths=[3,3,9,3]
        self.stem=nn.Sequential(
            nn.Conv1d(1,dims[0],kernel_size=4,stride=4),
            LayerNorm1D(dims[0])
        )
        self.stage1=nn.Sequential(*[ConvNeXtV2Block1D(dims[0]) for _ in range(depths[0])])
        self.down1=nn.Sequential(
            LayerNorm1D(dims[0]),
            nn.Conv1d(dims[0],dims[1],kernel_size=2,stride=2)
        )
        self.stage2=nn.Sequential(*[ConvNeXtV2Block1D(dims[1]) for _ in range(depths[1])])
        self.down2=nn.Sequential(
            LayerNorm1D(dims[1]),
            nn.Conv1d(dims[1],dims[2],kernel_size=2,stride=2)
        )
        self.stage3=nn.Sequential(*[ConvNeXtV2Block1D(dims[2]) for _ in range(depths[2])])
        self.down3=nn.Sequential(
            LayerNorm1D(dims[2]),
            nn.Conv1d(dims[2],dims[3],kernel_size=2,stride=2)
        )
        self.stage4=nn.Sequential(*[ConvNeXtV2Block1D(dims[3]) for _ in range(depths[3])])
        self.norm=nn.LayerNorm(dims[3])
        self.head=nn.Linear(dims[3],num_classes)
    def forward(self,x):
        x=self.stem(x)
        x=self.stage1(x)
        x=self.down1(x)
        x=self.stage2(x)
        x=self.down2(x)
        x=self.stage3(x)
        x=self.down3(x)
        x=self.stage4(x)
        x=x.mean(dim=-1)
        x=self.norm(x)
        logits=self.head(x)
        return logits
# ============================================================
# 10. CREATE MODEL
# ============================================================
model=ConvNeXtV2_1D(num_classes=2).to(DEVICE)
total_params=sum(p.numel() for p in model.parameters())
trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n========== MODEL ==========")
print("Total parameters:",f"{total_params:,}")
print("Trainable parameters:",f"{trainable_params:,}")
# ============================================================
# 11. FORWARD TEST
# ============================================================
model.eval()
with torch.no_grad():
    x_test=xb[:4].to(DEVICE,non_blocking=True)
    logits=model(x_test)
    probs=torch.softmax(logits,dim=1)
print("\n========== FORWARD TEST ==========")
print("Input shape:",x_test.shape)
print("Logits shape:",logits.shape)
print("Softmax shape:",probs.shape)
print("Example probabilities:")
print(probs.cpu())
print("\nCELL 1 READY")
