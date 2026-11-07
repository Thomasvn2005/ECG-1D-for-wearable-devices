
# ============================================================
# CELL 1 - ECG AF/AFL - 1D VISION TRANSFORMER (ViT-SMALL)
# TRAIN: PTB-XL + LTAFDB V3 ANY-AF
# VALIDATION: SHDB-AF
# TEST: MIT-AFDB
# ============================================================
import os
import sys
import subprocess
import random
from bisect import bisect_right
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from google.colab import drive
try:
    from einops import rearrange,repeat,pack,unpack
    from einops.layers.torch import Rearrange
except ImportError:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","einops"])
    from einops import rearrange,repeat,pack,unpack
    from einops.layers.torch import Rearrange
# ============================================================
# 1. MOUNT GOOGLE DRIVE
# ============================================================
drive.mount("/content/drive")
BASE_DIR="/content/drive/MyDrive/AF_ConvNeXtV2"
TRAIN_H5=BASE_DIR+"/preprocessed/PTBxl-processed/500hz processed/ptbxl_125hz_10s_leadI.h5"
LTAFDB_H5=BASE_DIR+"/preprocessed_v3/LTAFDB/ltafdb_125hz_10s_any_af.h5"
VAL_H5=BASE_DIR+"/preprocessed/SHDB-AF/shdb_af_125hz_10s_ECG1_labeled.h5"
TEST_H5=BASE_DIR+"/preprocessed/MIT-AFDB/mit_afdb_125hz_10s_ECG1_test.h5"
# ============================================================
# 2. CONFIGURATION
# ============================================================
SEED=42
FS=125
WINDOW_SEC=10
SEQ_LEN=1250
PATCH_SIZE=50
NUM_LEADS=1
NUM_CLASSES=2
BATCH_SIZE=128
NUM_WORKERS=0
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print("========== DEVICE ==========")
print("Device:",DEVICE)
if DEVICE.type=="cuda":
    print("GPU:",torch.cuda.get_device_name(0))
for path in [TRAIN_H5,LTAFDB_H5,VAL_H5,TEST_H5]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing HDF5 file: {path}")
print("All four HDF5 files found.")
# ============================================================
# 3. STANDARD HDF5 DATASET
# PTB-XL / SHDB-AF / MIT-AFDB
# ============================================================
class ECGH5Dataset(Dataset):
    def __init__(self,path,valid_labels_only=False):
        self.path=str(path)
        self.h5=None
        self.x=None
        self.y=None
        with h5py.File(self.path,"r") as f:
            assert "x" in f and "y" in f,f"Missing x/y in {self.path}"
            assert f["x"].shape[1:]==(NUM_LEADS,SEQ_LEN),f"Unexpected ECG shape in {self.path}"
            n=len(f["y"])
            assert len(f["x"])==n,"x/y length mismatch"
            if valid_labels_only:
                labels=np.asarray(f["y"][:])
                self.indices=np.where((labels==0)|(labels==1))[0].astype(np.int64)
            else:
                self.indices=np.arange(n,dtype=np.int64)
    def _open(self):
        if self.h5 is None:
            self.h5=h5py.File(self.path,"r")
            self.x=self.h5["x"]
            self.y=self.h5["y"]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,index):
        self._open()
        real_idx=int(self.indices[index])
        x=np.asarray(self.x[real_idx],dtype=np.float32)
        y=int(self.y[real_idx])
        return torch.from_numpy(x.copy()),torch.tensor(y,dtype=torch.long)
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
        except Exception:
            pass
# ============================================================
# 4. LTAFDB V3 GROUPED HDF5 DATASET
# STRUCTURE: 00/x,00/y,01/x,01/y,...
# ============================================================
class LTAFDBDataset(Dataset):
    def __init__(self,path):
        self.path=str(path)
        self.h5=None
        self.records=[]
        self.ends=[]
        self.total=0
        self.class_counts=np.zeros(2,dtype=np.int64)
        with h5py.File(self.path,"r") as f:
            assert str(f.attrs.get("version",""))=="v3","LTAFDB is not V3"
            assert bool(f.attrs.get("completed",False)),"LTAFDB V3 preprocessing incomplete"
            assert str(f.attrs.get("label_rule",""))=="v3_any_af_afib_afl_aflt_overlap_full_10s","Unexpected label rule"
            for name in sorted(f.keys()):
                group=f[name]
                assert isinstance(group,h5py.Group),f"Unexpected HDF5 entry: {name}"
                assert "x" in group and "y" in group,f"Missing x/y in record {name}"
                assert group["x"].shape[1:]==(NUM_LEADS,SEQ_LEN),f"Unexpected shape in {name}"
                n=len(group["y"])
                assert len(group["x"])==n,f"Length mismatch in {name}"
                assert bool(group.attrs.get("completed",False)),f"Incomplete record {name}"
                if n==0:
                    continue
                self.records.append(name)
                self.total+=n
                self.ends.append(self.total)
                labels=np.asarray(group["y"][:])
                assert np.all((labels==0)|(labels==1)),f"Invalid labels in {name}"
                self.class_counts+=np.bincount(labels.astype(np.int64),minlength=2)[:2]
        assert self.total>0,"No ECG windows found in LTAFDB"
    def _open(self):
        if self.h5 is None:
            self.h5=h5py.File(self.path,"r")
    def __len__(self):
        return self.total
    def __getitem__(self,index):
        self._open()
        index=int(index)
        if index<0:
            index+=self.total
        if index<0 or index>=self.total:
            raise IndexError(index)
        record_pos=bisect_right(self.ends,index)
        previous_end=0 if record_pos==0 else self.ends[record_pos-1]
        local_idx=index-previous_end
        group=self.h5[self.records[record_pos]]
        x=np.asarray(group["x"][local_idx],dtype=np.float32)
        y=int(group["y"][local_idx])
        return torch.from_numpy(x.copy()),torch.tensor(y,dtype=torch.long)
    def __getstate__(self):
        state=self.__dict__.copy()
        state["h5"]=None
        return state
    def __del__(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass
# ============================================================
# 5. LOAD DATASETS
# ============================================================
train_ptbxl=ECGH5Dataset(TRAIN_H5,valid_labels_only=True)
train_ltafdb=LTAFDBDataset(LTAFDB_H5)
train_dataset=ConcatDataset([train_ptbxl,train_ltafdb])
val_dataset=ECGH5Dataset(VAL_H5,valid_labels_only=True)
test_dataset=ECGH5Dataset(TEST_H5,valid_labels_only=False)
assert len(train_ltafdb.records)==84,"Expected 84 LTAFDB records"
assert len(train_ltafdb)==705787,"Unexpected LTAFDB window count"
assert int(train_ltafdb.class_counts[0])==329897,"Unexpected non-AF count"
assert int(train_ltafdb.class_counts[1])==375890,"Unexpected AF/AFL count"
print("\n========== DATASET SIZE ==========")
print("Train PTB-XL:",len(train_ptbxl))
print("Train LTAFDB:",len(train_ltafdb))
print("LTAFDB records:",len(train_ltafdb.records))
print("Train TOTAL:",len(train_dataset))
print("Validation SHDB-AF:",len(val_dataset))
print("Test MIT-AFDB:",len(test_dataset))
print("\n========== LTAFDB CLASS DISTRIBUTION ==========")
print("Non-AF:",int(train_ltafdb.class_counts[0]))
print("AF/AFL:",int(train_ltafdb.class_counts[1]))
print("AF percentage: %.2f%%"%(100*train_ltafdb.class_counts[1]/len(train_ltafdb)))
# ============================================================
# 6. DATA LOADERS
# ============================================================
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=(DEVICE.type=="cuda"),drop_last=False)
val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=(DEVICE.type=="cuda"),drop_last=False)
test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=(DEVICE.type=="cuda"),drop_last=False)
print("\nFull train batches:",len(train_loader))
# ============================================================
# 7. DROP PATH
# ============================================================
class DropPath(nn.Module):
    def __init__(self,drop_prob=0.,scale_by_keep=True):
        super().__init__()
        self.drop_prob=drop_prob
        self.scale_by_keep=scale_by_keep
    def forward(self,x):
        if self.drop_prob<=0. or not self.training:
            return x
        keep_prob=1-self.drop_prob
        shape=(x.shape[0],)+(1,)*(x.ndim-1)
        random_tensor=x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob>0. and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x*random_tensor
# ============================================================
# 8. PRE NORM
# ============================================================
class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)
# ============================================================
# 9. FEED FORWARD
# ============================================================
class FeedForward(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,drop_out_rate=0.):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_dim,output_dim),
            nn.Dropout(drop_out_rate)
        )
    def forward(self,x):
        return self.net(x)
# ============================================================
# 10. MULTI-HEAD SELF ATTENTION
# ============================================================
class Attention(nn.Module):
    def __init__(self,input_dim,output_dim,heads=8,dim_head=64,qkv_bias=True,drop_out_rate=0.,attn_drop_out_rate=0.):
        super().__init__()
        inner_dim=dim_head*heads
        project_out=not(heads==1 and dim_head==input_dim)
        self.heads=heads
        self.scale=dim_head**-0.5
        self.attend=nn.Softmax(dim=-1)
        self.dropout=nn.Dropout(attn_drop_out_rate)
        self.to_qkv=nn.Linear(input_dim,inner_dim*3,bias=qkv_bias)
        if project_out:
            self.to_out=nn.Sequential(nn.Linear(inner_dim,output_dim),nn.Dropout(drop_out_rate))
        else:
            self.to_out=nn.Identity()
    def forward(self,x):
        qkv=self.to_qkv(x).chunk(3,dim=-1)
        q,k,v=map(lambda t:rearrange(t,"b n (h d) -> b h n d",h=self.heads),qkv)
        dots=torch.matmul(q,k.transpose(-1,-2))*self.scale
        attn=self.attend(dots)
        attn=self.dropout(attn)
        out=torch.matmul(attn,v)
        out=rearrange(out,"b h n d -> b n (h d)")
        return self.to_out(out)
# ============================================================
# 11. TRANSFORMER BLOCK
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,heads=8,dim_head=32,qkv_bias=True,drop_out_rate=0.,attn_drop_out_rate=0.,drop_path_rate=0.):
        super().__init__()
        assert input_dim==output_dim,"Residual connection requires equal dimensions"
        self.attn=PreNorm(input_dim,Attention(input_dim,output_dim,heads,dim_head,qkv_bias,drop_out_rate,attn_drop_out_rate))
        self.droppath1=DropPath(drop_path_rate) if drop_path_rate>0 else nn.Identity()
        self.ff=PreNorm(output_dim,FeedForward(output_dim,output_dim,hidden_dim,drop_out_rate))
        self.droppath2=DropPath(drop_path_rate) if drop_path_rate>0 else nn.Identity()
    def forward(self,x):
        x=self.droppath1(self.attn(x))+x
        x=self.droppath2(self.ff(x))+x
        return x
# ============================================================
# 12. 1D VISION TRANSFORMER
# ============================================================
class ViT(nn.Module):
    def __init__(self,seq_len,patch_size,num_leads,num_classes=2,width=384,depth=12,mlp_dim=1536,heads=6,dim_head=64,qkv_bias=True,drop_out_rate=0.,attn_drop_out_rate=0.,drop_path_rate=0.):
        super().__init__()
        assert seq_len%patch_size==0,"Sequence length must be divisible by patch size"
        self.width=width
        self.depth=depth
        num_patches=seq_len//patch_size
        patch_dim=num_leads*patch_size
        self.to_patch_embedding=nn.Sequential(
            Rearrange("b c (n p) -> b n (p c)",p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim,width),
            nn.LayerNorm(width)
        )
        self.pos_embedding=nn.Parameter(torch.randn(1,num_patches+1,width))
        self.cls_embedding=nn.Parameter(torch.randn(width))
        drop_rates=torch.linspace(0,drop_path_rate,depth).tolist()
        for i in range(depth):
            block=TransformerBlock(width,width,mlp_dim,heads,dim_head,qkv_bias,drop_out_rate,attn_drop_out_rate,drop_rates[i])
            self.add_module(f"block{i}",block)
        self.dropout=nn.Dropout(drop_out_rate)
        self.norm=nn.LayerNorm(width)
        self.head=nn.Identity() if num_classes is None else nn.Linear(width,num_classes)
    def forward_encoding(self,series):
        x=self.to_patch_embedding(series)
        b,n,_=x.shape
        cls_embeddings=repeat(self.cls_embedding,"d -> b d",b=b)
        x,ps=pack([cls_embeddings,x],"b * d")
        x=x+self.pos_embedding[:,:n+1]
        x=self.dropout(x)
        for i in range(self.depth):
            x=getattr(self,f"block{i}")(x)
        cls_embeddings,_=unpack(x,ps,"b * d")
        return self.norm(cls_embeddings)
    def forward(self,series):
        return self.head(self.forward_encoding(series))
# ============================================================
# 13. VIT SMALL
# ============================================================
def vit_small(num_leads=1,num_classes=2,seq_len=1250,patch_size=50,**kwargs):
    return ViT(
        seq_len=seq_len,
        patch_size=patch_size,
        num_leads=num_leads,
        num_classes=num_classes,
        width=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        **kwargs
    )
# ============================================================
# 14. INITIALIZE MODEL
# ============================================================
model=vit_small(
    num_leads=NUM_LEADS,
    num_classes=NUM_CLASSES,
    seq_len=SEQ_LEN,
    patch_size=PATCH_SIZE
).to(DEVICE)
total_params=sum(p.numel() for p in model.parameters())
trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n========== MODEL ==========")
print("Architecture: 1D ViT-Small")
print("Total parameters:",f"{total_params:,}")
print("Trainable parameters:",f"{trainable_params:,}")
print("Patch size:",PATCH_SIZE)
print("Number of patches:",SEQ_LEN//PATCH_SIZE)
print("Tokens including CLS:",SEQ_LEN//PATCH_SIZE+1)
# ============================================================
# 15. TRAIN BATCH CHECK
# ============================================================
x_batch,y_batch=next(iter(train_loader))
print("\n========== TRAIN BATCH CHECK ==========")
print("x batch shape:",x_batch.shape)
print("y batch shape:",y_batch.shape)
print("x dtype:",x_batch.dtype)
print("y dtype:",y_batch.dtype)
print("Labels in batch:",torch.unique(y_batch))
assert x_batch.shape[1:]==(NUM_LEADS,SEQ_LEN),"Invalid ECG input shape"
assert torch.isfinite(x_batch).all(),"NaN or Inf in ECG batch"
assert torch.all((y_batch==0)|(y_batch==1)),"Invalid training labels"
print("TRAIN BATCH CHECK: PASS")
# ============================================================
# 16. FORWARD PASS CHECK
# ============================================================
model.eval()
with torch.no_grad():
    sample=x_batch[:4].to(DEVICE)
    output=model(sample)
    probabilities=torch.softmax(output,dim=1)
print("\n========== FORWARD PASS CHECK ==========")
print("Input shape:",tuple(sample.shape))
print("Output shape:",tuple(output.shape))
print("Probabilities:")
print(probabilities.cpu())
assert output.shape==(4,NUM_CLASSES),"Incorrect classifier output shape"
assert torch.isfinite(output).all(),"Non-finite model output"
print("FORWARD PASS: PASS")
print("\n========== CELL 1 READY ==========")
