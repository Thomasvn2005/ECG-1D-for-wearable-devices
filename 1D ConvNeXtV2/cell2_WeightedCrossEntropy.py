
import os,time,math,random,json,h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,precision_score,recall_score
EPOCHS=3
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
SEED=42
TRAIN_H5="/content/drive/MyDrive/AF_ConvNeXtV2/preprocessed/PTBxl-processed/500hz processed/ptbxl_125hz_10s_leadI.h5"
SAVE_DIR="/content/drive/MyDrive/AF_ConvNeXtV2"
BEST_MODEL_PATH=os.path.join(SAVE_DIR,"best_model_weighted_v2.pth")
FINAL_MODEL_PATH=os.path.join(SAVE_DIR,"final_model_epoch20_weighted_v2.pth")
HISTORY_PATH=os.path.join(SAVE_DIR,"training_history_weighted_v2.json")
os.makedirs(SAVE_DIR,exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
assert "ConvNeXtV2_1D" in globals(),"Please run Cell 1 first"
assert "train_loader" in globals(),"train_loader not found"
assert "val_loader" in globals(),"val_loader not found"
assert os.path.exists(TRAIN_H5),"PTB-XL training file not found"
model=ConvNeXtV2_1D(num_classes=2).to(DEVICE)
print("========== CELL 2 CONFIG ==========")
print("Device:",DEVICE)
print("Epochs:",EPOCHS)
print("Optimizer: AdamW")
print("Learning rate:",LEARNING_RATE)
print("Weight decay:",WEIGHT_DECAY)
print("Fresh model initialized: YES")
print("Best model:",BEST_MODEL_PATH)
print("Final model:",FINAL_MODEL_PATH)
print("\n========== CLASS DISTRIBUTION ==========")
with h5py.File(TRAIN_H5,"r") as f:
    all_labels=np.asarray(f["y"][:]).reshape(-1)
if "train_dataset" in globals() and hasattr(train_dataset,"indices"):
    train_indices=np.asarray(train_dataset.indices)
    train_labels=all_labels[train_indices]
else:
    train_labels=all_labels
class_counts_np=np.bincount(train_labels.astype(np.int64),minlength=2)
assert len(class_counts_np)==2 and np.all(class_counts_np>0),"Both classes are required"
class_counts=torch.tensor(class_counts_np,dtype=torch.float32)
class_weights=(class_counts.sum()/(2.0*class_counts)).to(DEVICE)
print("Non-AF samples:",int(class_counts_np[0]))
print("AF samples:",int(class_counts_np[1]))
print("Class imbalance ratio:",round(class_counts_np[0]/class_counts_np[1],4))
print("Class weights:",class_weights.detach().cpu().tolist())
criterion=nn.CrossEntropyLoss(weight=class_weights)
optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
def colored_noise(length,kind,device,dtype):
    n=torch.randn(length,device=device,dtype=dtype)
    if kind!="gaussian":
        f=torch.fft.rfft(n)
        freq=torch.arange(f.shape[-1],device=device,dtype=dtype)
        freq[0]=1.0
        if kind=="pink":
            f=f/torch.sqrt(freq)
        elif kind=="brown":
            f=f/freq
        n=torch.fft.irfft(f,n=length)
    n=n-n.mean()
    n=n/(n.std(unbiased=False)+1e-6)
    return n
def augment_batch(x,y):
    x=x.clone()
    B,C,L=x.shape
    t=torch.arange(L,device=x.device,dtype=x.dtype)/125.0
    for i in range(B):
        if random.random()<0.30:
            x[i]*=random.uniform(0.85,1.15)
        if random.random()<0.20:
            x[i]+=random.uniform(-0.03,0.03)
        if random.random()<0.30:
            kind=random.choice(["gaussian","pink","brown"])
            noise=colored_noise(L,kind,x.device,x.dtype)
            signal_std=x[i,0].std(unbiased=False).detach()
            noise_factor=random.uniform(0.005,0.05)
            x[i,0]+=noise_factor*signal_std*noise
        if random.random()<0.30:
            shift=random.randint(-62,62)
            x[i]=torch.roll(x[i],shifts=shift,dims=-1)
        if random.random()<0.15:
            freq=random.uniform(0.05,0.3)
            amp=random.uniform(0.0,0.03)
            phase=random.uniform(0.0,2.0*math.pi)
            x[i,0]+=amp*torch.sin(2.0*math.pi*freq*t+phase)
        if random.random()<0.05:
            x[i]*=-1.0
    return x
def train_one_epoch(model,loader,criterion,optimizer,device,epoch):
    model.train()
    running_loss=0.0
    total=0
    correct=0
    y_true=[]
    y_pred=[]
    t0=time.time()
    for batch_idx,(x,y) in enumerate(loader,1):
        x=x.to(device,non_blocking=True)
        y=y.to(device,non_blocking=True).long()
        x=augment_batch(x,y)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x)
        loss=criterion(logits,y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}, batch {batch_idx}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimizer.step()
        running_loss+=loss.item()*x.size(0)
        pred=torch.argmax(logits.detach(),dim=1)
        correct+=(pred==y).sum().item()
        total+=y.size(0)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        if batch_idx%25==0 or batch_idx==len(loader):
            print(f"\rEpoch {epoch} TRAIN {batch_idx}/{len(loader)} | loss={running_loss/total:.5f} | acc={correct/total:.4f}",end="")
    print()
    y_true=np.concatenate(y_true)
    y_pred=np.concatenate(y_pred)
    metrics={"loss":running_loss/total,"accuracy":accuracy_score(y_true,y_pred),"precision":precision_score(y_true,y_pred,zero_division=0),"recall":recall_score(y_true,y_pred,zero_division=0),"f1":f1_score(y_true,y_pred,zero_division=0),"time":time.time()-t0}
    return metrics
@torch.no_grad()
def validate(model,loader,criterion,device):
    model.eval()
    running_loss=0.0
    total=0
    y_true=[]
    y_prob=[]
    t0=time.time()
    for batch_idx,(x,y) in enumerate(loader,1):
        x=x.to(device,non_blocking=True)
        y=y.to(device,non_blocking=True).long()
        if not torch.all((y==0)|(y==1)):
            raise ValueError("Validation contains invalid labels. Exclude SHDB-AF label -1 in Cell 1.")
        logits=model(x)
        loss=criterion(logits,y)
        prob=torch.softmax(logits,dim=1)[:,1]
        running_loss+=loss.item()*x.size(0)
        total+=y.size(0)
        y_true.append(y.cpu().numpy())
        y_prob.append(prob.cpu().numpy())
        if batch_idx%500==0 or batch_idx==len(loader):
            print(f"\rVALIDATION {batch_idx}/{len(loader)}",end="")
    print()
    y_true=np.concatenate(y_true)
    y_prob=np.concatenate(y_prob)
    y_pred=(y_prob>=0.5).astype(np.uint8)
    val_auc=roc_auc_score(y_true,y_prob) if len(np.unique(y_true))==2 else float("nan")
    metrics={"loss":running_loss/total,"accuracy":accuracy_score(y_true,y_pred),"precision":precision_score(y_true,y_pred,zero_division=0),"recall":recall_score(y_true,y_pred,zero_division=0),"f1":f1_score(y_true,y_pred,zero_division=0),"auc":val_auc,"prob_mean":float(y_prob.mean()),"prob_std":float(y_prob.std()),"prob_min":float(y_prob.min()),"prob_max":float(y_prob.max()),"predicted_af":int(y_pred.sum()),"time":time.time()-t0}
    return metrics
def create_checkpoint(model,optimizer,epoch,train_metrics,val_metrics):
    return {"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"train_loss":train_metrics["loss"],"train_accuracy":train_metrics["accuracy"],"train_precision":train_metrics["precision"],"train_recall":train_metrics["recall"],"train_f1":train_metrics["f1"],"val_loss":val_metrics["loss"],"val_accuracy":val_metrics["accuracy"],"val_precision":val_metrics["precision"],"val_recall":val_metrics["recall"],"val_f1":val_metrics["f1"],"val_auc":val_metrics["auc"],"val_prob_mean":val_metrics["prob_mean"],"val_prob_std":val_metrics["prob_std"],"class_counts":class_counts_np.tolist(),"class_weights":class_weights.detach().cpu().tolist(),"learning_rate":LEARNING_RATE,"weight_decay":WEIGHT_DECAY,"loss_function":"WeightedCrossEntropyLoss","model_parameters":sum(p.numel() for p in model.parameters()),"input_fs":125,"window_sec":10,"window_samples":1250,"classes":["non-AF/AFL","AF/AFL"]}
print("\n========== TRAINING START ==========")
history=[]
best_val_auc=-float("inf")
best_epoch=0
for epoch in range(1,EPOCHS+1):
    print("\n"+"="*70)
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("="*70)
    train_metrics=train_one_epoch(model,train_loader,criterion,optimizer,DEVICE,epoch)
    print(f"TRAIN | loss={train_metrics['loss']:.6f} | accuracy={train_metrics['accuracy']:.4f} | F1={train_metrics['f1']:.4f} | recall={train_metrics['recall']:.4f} | time={train_metrics['time']/60:.1f} min")
    print("\nRunning SHDB-AF validation...")
    val_metrics=validate(model,val_loader,criterion,DEVICE)
    print("\n========== VALIDATION RESULTS ==========")
    print(f"VAL loss      : {val_metrics['loss']:.6f}")
    print(f"VAL accuracy  : {val_metrics['accuracy']:.4f}")
    print(f"VAL precision : {val_metrics['precision']:.4f}")
    print(f"VAL recall    : {val_metrics['recall']:.4f}")
    print(f"VAL F1        : {val_metrics['f1']:.4f}")
    print(f"VAL AUC       : {val_metrics['auc']:.4f}")
    print(f"VAL P(AF)     : mean={val_metrics['prob_mean']:.6f} | std={val_metrics['prob_std']:.6f}")
    print("Predicted AF windows:",val_metrics["predicted_af"])
    print(f"VAL time      : {val_metrics['time']/60:.1f} min")
    history.append({"epoch":epoch,"train":train_metrics,"validation":val_metrics})
    checkpoint=create_checkpoint(model,optimizer,epoch,train_metrics,val_metrics)
    current_auc=val_metrics["auc"]
    if np.isfinite(current_auc) and current_auc>best_val_auc:
        best_val_auc=current_auc
        best_epoch=epoch
        torch.save(checkpoint,BEST_MODEL_PATH)
        print("\nNEW BEST MODEL SAVED!")
        print("Best epoch:",best_epoch)
        print("Best validation AUC:",best_val_auc)
        print("Path:",BEST_MODEL_PATH)
    torch.save(checkpoint,FINAL_MODEL_PATH)
    with open(HISTORY_PATH,"w",encoding="utf-8") as f:
        json.dump(history,f,indent=2,allow_nan=True)
    print("\nLatest checkpoint saved.")
    print("Current best epoch:",best_epoch)
    print("Current best AUC:",best_val_auc)
    if val_metrics["prob_std"]<0.001:
        print("\nWARNING: PREDICTION COLLAPSE SUSPECTED!")
        print("Validation probabilities are nearly constant.")
        if epoch>=3:
            print("If AUC remains near 0.5, investigate labels and feature extraction before continuing.")
print("\n"+"="*70)
print("TRAINING FINISHED")
print("="*70)
print("Total epochs:",EPOCHS)
print("Best epoch:",best_epoch)
print("Best validation AUC:",best_val_auc)
print("\nBEST MODEL:",BEST_MODEL_PATH)
print("FINAL EPOCH MODEL:",FINAL_MODEL_PATH)
print("TRAINING HISTORY:",HISTORY_PATH)
print("\n========== CELL 2 FINISHED ==========")
