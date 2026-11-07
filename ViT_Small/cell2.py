
# ============================================================
# CELL 2 - TRAINING 1D ViT-SMALL FOR ECG AF/AFL
# TRAIN: PTB-XL + LTAFDB V3
# VALIDATION: SHDB-AF
# CHECKPOINT + RESUME + BEST MODEL + FINAL MODEL
# ============================================================
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from tqdm.auto import tqdm
# ============================================================
# 1. CONFIGURATION
# ============================================================
SEED=42
EPOCHS=20
LEARNING_RATE=3e-4
WEIGHT_DECAY=1e-4
VAL_SAMPLES_PER_EPOCH=20000
USE_AUGMENTATION=True
RESUME=True
RUN_FULL_VALIDATION_AT_END=True
PRINT_EVERY=100
SAVE_DIR="/content/drive/MyDrive/AF_ConvNeXtV2/ViT_Small"
os.makedirs(SAVE_DIR,exist_ok=True)
LATEST_PATH=os.path.join(SAVE_DIR,"latest_vit_small_ptbxl_ltafdb_v3.pth")
BEST_PATH=os.path.join(SAVE_DIR,"best_vit_small_ptbxl_ltafdb_v3.pth")
FINAL_PATH=os.path.join(SAVE_DIR,"final_vit_small_ptbxl_ltafdb_v3.pth")
HISTORY_PATH=os.path.join(SAVE_DIR,"training_history_vit_small.json")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(DEVICE)
print("========== TRAINING CONFIGURATION ==========")
print("Model: ViT-Small 1D")
print("Device:",DEVICE)
if DEVICE.type=="cuda":
    print("GPU:",torch.cuda.get_device_name(0))
else:
    print("WARNING: CPU training will be very slow.")
print("Epochs:",EPOCHS)
print("Batch size:",train_loader.batch_size)
print("Train samples:",len(train_dataset))
print("Validation samples:",len(val_dataset))
print("Checkpoint folder:",SAVE_DIR)
# ============================================================
# 2. REPRODUCIBILITY
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# ============================================================
# 3. ECG DATA AUGMENTATION
# ============================================================
def augment_batch(x,y):
    x=x.clone()
    b,c,l=x.shape
    device=x.device
    if random.random()<0.75:
        scale=torch.empty(b,1,1,device=device).uniform_(0.6,1.4)
        x=x*scale
    if random.random()<0.75:
        baseline=torch.empty(b,1,1,device=device).uniform_(-0.2,0.2)
        x=x+baseline
    if random.random()<0.75:
        noise_type=random.choice(["gaussian","pink","brown"])
        noise=torch.randn_like(x)
        if noise_type=="pink":
            noise=torch.cumsum(noise,dim=-1)
            noise=noise-torch.mean(noise,dim=-1,keepdim=True)
        elif noise_type=="brown":
            noise=torch.cumsum(torch.cumsum(noise,dim=-1),dim=-1)
            noise=noise-torch.mean(noise,dim=-1,keepdim=True)
        noise=noise/(noise.std(dim=-1,keepdim=True)+1e-6)
        amplitude=torch.empty(b,1,1,device=device).uniform_(-0.2,0.2)
        x=x+amplitude*noise
    if random.random()<0.75:
        shifts=torch.randint(-l//2,l//2+1,(b,),device=device)
        positions=(torch.arange(l,device=device).unsqueeze(0)-shifts.unsqueeze(1))%l
        x=torch.gather(x,2,positions.unsqueeze(1).expand(-1,c,-1))
    if random.random()<0.25:
        t=torch.arange(l,device=device,dtype=x.dtype)/FS
        freq=torch.empty(b,1,1,device=device).uniform_(0.05,0.5)
        amplitude=torch.empty(b,1,1,device=device).uniform_(0,0.2)
        phase=torch.empty(b,1,1,device=device).uniform_(0,2*np.pi)
        artifact=amplitude*torch.sin(2*np.pi*freq*t.view(1,1,-1)+phase)
        x=x+artifact
    if random.random()<0.20:
        mask=torch.rand(b,1,1,device=device)<0.20
        x=torch.where(mask,-x,x)
    if random.random()<0.05:
        mask=(y==0)&(torch.rand(b,device=device)<0.05)
        if mask.any():
            noise=torch.randn_like(x[mask])
            noise=noise/(noise.std(dim=-1,keepdim=True)+1e-6)
            scale=torch.empty(int(mask.sum().item()),1,1,device=device).uniform_(0.05,0.3)
            x[mask]=noise*scale
    return x
# ============================================================
# 4. OPTIMIZER AND LOSS
# ============================================================
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
# ============================================================
# 5. FIXED VALIDATION SUBSET
# ============================================================
rng=np.random.default_rng(SEED)
val_count=min(VAL_SAMPLES_PER_EPOCH,len(val_dataset))
val_indices=np.sort(rng.choice(len(val_dataset),size=val_count,replace=False))
val_subset=Subset(val_dataset,val_indices.tolist())
val_subset_loader=DataLoader(val_subset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=(DEVICE.type=="cuda"),drop_last=False)
print("Validation subset:",len(val_subset))
# ============================================================
# 6. VALIDATION FUNCTION
# ============================================================
@torch.no_grad()
def evaluate_model(model,loader,description="Validation"):
    model.eval()
    total_loss=0.0
    total_samples=0
    all_true=[]
    all_pred=[]
    all_prob=[]
    progress=tqdm(loader,desc=description,leave=False)
    for x,y in progress:
        x=x.to(DEVICE,non_blocking=True)
        y=y.to(DEVICE,non_blocking=True)
        logits=model(x)
        loss=criterion(logits,y)
        probs=torch.softmax(logits,dim=1)[:,1]
        preds=torch.argmax(logits,dim=1)
        total_loss+=loss.item()*y.size(0)
        total_samples+=y.size(0)
        all_true.extend(y.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_prob.extend(probs.cpu().numpy())
    all_true=np.asarray(all_true,dtype=np.int64)
    all_pred=np.asarray(all_pred,dtype=np.int64)
    all_prob=np.asarray(all_prob,dtype=np.float32)
    metrics={
        "loss":float(total_loss/max(total_samples,1)),
        "accuracy":float(accuracy_score(all_true,all_pred)),
        "precision":float(precision_score(all_true,all_pred,zero_division=0)),
        "recall":float(recall_score(all_true,all_pred,zero_division=0)),
        "f1":float(f1_score(all_true,all_pred,zero_division=0)),
        "auc":float(roc_auc_score(all_true,all_prob)) if len(np.unique(all_true))==2 else None,
        "samples":int(total_samples)
    }
    return metrics
# ============================================================
# 7. CHECKPOINT FUNCTIONS
# ============================================================
history=[]
best_auc=-1.0
start_epoch=1
def save_checkpoint(path,epoch,train_metrics,val_metrics,best_auc,history):
    checkpoint={
        "epoch":int(epoch),
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "train_metrics":train_metrics,
        "val_metrics":val_metrics,
        "best_auc":float(best_auc),
        "history":history,
        "architecture":"ViT-Small-1D",
        "input_fs":FS,
        "window_sec":WINDOW_SEC,
        "window_samples":SEQ_LEN,
        "patch_size":PATCH_SIZE,
        "num_classes":NUM_CLASSES,
        "classes":["non-AF/AFL","AF/AFL"],
        "train_datasets":["PTB-XL","LTAFDB-V3"],
        "validation_dataset":"SHDB-AF",
        "seed":SEED
    }
    temp_path=path+".tmp"
    torch.save(checkpoint,temp_path)
    os.replace(temp_path,path)
# ============================================================
# 8. RESUME TRAINING
# ============================================================
if RESUME and os.path.isfile(LATEST_PATH):
    print("\n========== RESUME CHECKPOINT ==========")
    checkpoint=torch.load(LATEST_PATH,map_location=DEVICE,weights_only=False)
    if checkpoint.get("architecture")!="ViT-Small-1D":
        raise ValueError("Checkpoint architecture does not match ViT-Small.")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch=int(checkpoint["epoch"])+1
    best_auc=float(checkpoint.get("best_auc",-1.0))
    history=checkpoint.get("history",[])
    print("Resuming from completed epoch:",start_epoch-1)
    print("Next epoch:",start_epoch)
    print("Best validation AUC:",best_auc)
else:
    print("\n========== NEW TRAINING ==========")
    print("Starting from epoch 1.")
# ============================================================
# 9. TRAINING LOOP
# ============================================================
training_start=time.time()
for epoch in range(start_epoch,EPOCHS+1):
    epoch_start=time.time()
    model.train()
    running_loss=0.0
    running_correct=0
    running_samples=0
    progress=tqdm(train_loader,desc=f"Epoch {epoch}/{EPOCHS}")
    for batch_idx,(x,y) in enumerate(progress,1):
        x=x.to(DEVICE,non_blocking=True)
        y=y.to(DEVICE,non_blocking=True)
        if USE_AUGMENTATION:
            x=augment_batch(x,y)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x)
        loss=criterion(logits,y)
        loss.backward()
        optimizer.step()
        batch_size=y.size(0)
        running_loss+=loss.item()*batch_size
        running_correct+=(logits.argmax(dim=1)==y).sum().item()
        running_samples+=batch_size
        if batch_idx%PRINT_EVERY==0 or batch_idx==len(train_loader):
            avg_loss=running_loss/max(running_samples,1)
            avg_acc=running_correct/max(running_samples,1)
            progress.set_postfix(loss=f"{avg_loss:.5f}",accuracy=f"{avg_acc:.4f}")
    train_metrics={
        "loss":float(running_loss/max(running_samples,1)),
        "accuracy":float(running_correct/max(running_samples,1)),
        "samples":int(running_samples)
    }
    print(f"\n========== EPOCH {epoch} TRAIN ==========")
    print("Train loss:",round(train_metrics["loss"],6))
    print("Train accuracy:",round(train_metrics["accuracy"],6))
    val_metrics=evaluate_model(model,val_subset_loader,description=f"Validation epoch {epoch}")
    print(f"========== EPOCH {epoch} VALIDATION ==========")
    for key,value in val_metrics.items():
        print(f"{key}: {value}")
    current_auc=val_metrics["auc"]
    improved=current_auc is not None and current_auc>best_auc
    if improved:
        best_auc=current_auc
    epoch_seconds=time.time()-epoch_start
    epoch_result={
        "epoch":epoch,
        "train":train_metrics,
        "validation":val_metrics,
        "epoch_seconds":float(epoch_seconds)
    }
    history.append(epoch_result)
    save_checkpoint(LATEST_PATH,epoch,train_metrics,val_metrics,best_auc,history)
    if improved:
        save_checkpoint(BEST_PATH,epoch,train_metrics,val_metrics,best_auc,history)
        print("NEW BEST MODEL SAVED:",BEST_PATH)
    with open(HISTORY_PATH,"w",encoding="utf-8") as f:
        json.dump(history,f,indent=2,ensure_ascii=False)
    print("Latest checkpoint saved:",LATEST_PATH)
    print("Epoch duration:",round(epoch_seconds/60,2),"minutes")
    print("Best validation AUC:",round(best_auc,6))
# ============================================================
# 10. FULL VALIDATION AT THE END
# ============================================================
if RUN_FULL_VALIDATION_AT_END:
    print("\n========== FULL SHDB-AF VALIDATION ==========")
    if os.path.isfile(BEST_PATH):
        best_checkpoint=torch.load(BEST_PATH,map_location=DEVICE,weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        final_epoch=int(best_checkpoint["epoch"])
        print("Loaded best model from epoch:",final_epoch)
    elif os.path.isfile(LATEST_PATH):
        latest_checkpoint=torch.load(LATEST_PATH,map_location=DEVICE,weights_only=False)
        model.load_state_dict(latest_checkpoint["model_state_dict"])
        final_epoch=int(latest_checkpoint["epoch"])
        print("Loaded latest model from epoch:",final_epoch)
    else:
        final_epoch=0
    full_val_metrics=evaluate_model(model,val_loader,description="Full SHDB-AF validation")
    print("\n========== FULL VALIDATION RESULTS ==========")
    for key,value in full_val_metrics.items():
        print(f"{key}: {value}")
    save_checkpoint(FINAL_PATH,final_epoch,{},full_val_metrics,best_auc,history)
    print("Final model saved:",FINAL_PATH)
# ============================================================
# 11. FINISHED
# ============================================================
total_seconds=time.time()-training_start
print("\n========== TRAINING COMPLETED ==========")
print("Total session time:",round(total_seconds/3600,2),"hours")
print("Latest checkpoint:",LATEST_PATH)
print("Best checkpoint:",BEST_PATH)
print("Final checkpoint:",FINAL_PATH)
print("Training history:",HISTORY_PATH)
