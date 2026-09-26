import os,time,math,random 
import numpy as np 
import torch 
import torch.nn as nn 
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,precision_score,recall_score 
EPOCHS=20 
LEARNING_RATE=1e-3 
WEIGHT_DECAY=1e-4 
FINAL_MODEL_PATH="/content/drive/MyDrive/AF_ConvNeXtV2/final_model_epoch20.pth" 
os.makedirs(os.path.dirname(FINAL_MODEL_PATH),exist_ok=True) 
criterion=nn.CrossEntropyLoss() 
optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY) 
print("========== CELL 2 CONFIG ==========") 
print("Device:",DEVICE) 
print("Epochs:",EPOCHS) 
print("Optimizer: AdamW") 
print("Learning rate:",LEARNING_RATE) 
print("Weight decay:",WEIGHT_DECAY) 
print("Loss: CrossEntropyLoss") 
print("Final model:",FINAL_MODEL_PATH) 
def colored_noise(length,kind,device,dtype): 
    if kind=="gaussian": 
        n=torch.randn(length,device=device,dtype=dtype) 
    else: 
        n=torch.randn(length,device=device,dtype=dtype) 
        f=torch.fft.rfft(n) 
        freq=torch.arange(f.shape[-1],device=device,dtype=dtype) 
        freq[0]=1.0 
        if kind=="pink": 
            f=f/torch.sqrt(freq) 
        elif kind=="brown": 
            f=f/freq 
        n=torch.fft.irfft(f,n=length) 
        n=n/(n.std()+1e-6) 
    return n 
def augment_batch(x,y): 
    x=x.clone() 
    B,C,L=x.shape 
    for i in range(B): 
        if random.random()<0.75: 
            x[i]*=random.uniform(0.6,1.4) 
        if random.random()<0.75: 
            x[i]+=random.uniform(-0.2,0.2) 
        if random.random()<0.75: 
            kind=random.choice(["gaussian","pink","brown"]) 
            noise=colored_noise(L,kind,x.device,x.dtype) 
            x[i,0]+=random.uniform(-0.2,0.2)*noise 
        if random.random()<0.75: 
            shift=random.randint(-625,625) 
            x[i]=torch.roll(x[i],shifts=shift,dims=-1) 
        if random.random()<0.25: 
            t=torch.arange(L,device=x.device,dtype=x.dtype)/125.0 
            freq=random.uniform(0.05,0.5) 
            amp=random.uniform(0.0,0.2) 
            phase=random.uniform(0.0,2.0*math.pi) 
            x[i,0]+=amp*torch.sin(2.0*math.pi*freq*t+phase) 
        if random.random()<0.20: 
            x[i]*=-1.0 
        if int(y[i].item())==0 and random.random()<0.05: 
            kind=random.choice(["gaussian","pink","brown"]) 
            noise=colored_noise(L,kind,x.device,x.dtype) 
            x[i,0]=random.uniform(0.05,0.3)*noise 
    return x 
def train_one_epoch(model,loader,criterion,optimizer,device,epoch): 
    model.train() 
    running_loss=0.0 
    total=0 
    correct=0 
    t0=time.time() 
    for batch_idx,(x,y) in enumerate(loader,1): 
        x=x.to(device,non_blocking=True) 
        y=y.to(device,non_blocking=True) 
        x=augment_batch(x,y) 
        optimizer.zero_grad(set_to_none=True) 
        logits=model(x) 
        loss=criterion(logits,y) 
        loss.backward() 
        optimizer.step() 
        running_loss+=loss.item()*x.size(0) 
        pred=torch.argmax(logits,dim=1) 
        correct+=(pred==y).sum().item() 
        total+=y.size(0) 
        if batch_idx%25==0 or batch_idx==len(loader): 
            print(f"\rEpoch {epoch} TRAIN {batch_idx}/{len(loader)} | loss={running_loss/total:.5f} | acc={correct/total:.4f}",end="") 
    print() 
    return running_loss/total,correct/total,time.time()-t0 
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
        y=y.to(device,non_blocking=True) 
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
    val_loss=running_loss/total 
    val_acc=accuracy_score(y_true,y_pred) 
    val_precision=precision_score(y_true,y_pred,zero_division=0) 
    val_recall=recall_score(y_true,y_pred,zero_division=0) 
    val_f1=f1_score(y_true,y_pred,zero_division=0) 
    val_auc=roc_auc_score(y_true,y_prob) 
    return val_loss,val_acc,val_precision,val_recall,val_f1,val_auc,time.time()-t0 
print("\n========== TRAINING START ==========") 
history=[] 
for epoch in range(1,EPOCHS+1): 
    print("\n"+"="*70) 
    print(f"EPOCH {epoch}/{EPOCHS}") 
    print("="*70) 
    train_loss,train_acc,train_time=train_one_epoch(model,train_loader,criterion,optimizer,DEVICE,epoch) 
    history.append({"epoch":epoch,"train_loss":train_loss,"train_acc":train_acc}) 
    print(f"TRAIN | loss={train_loss:.6f} | accuracy={train_acc:.4f} | time={train_time/60:.1f} min") 
    checkpoint={ 
        "epoch":epoch, 
        "model_state_dict":model.state_dict(), 
        "optimizer_state_dict":optimizer.state_dict(), 
        "train_loss":train_loss, 
        "train_accuracy":train_acc, 
        "model_parameters":sum(p.numel() for p in model.parameters()), 
        "input_fs":125, 
        "window_sec":10, 
        "window_samples":1250, 
        "classes":["non-AF/AFL","AF/AFL"] 
    } 
    torch.save(checkpoint,FINAL_MODEL_PATH) 
    print("Checkpoint saved:",FINAL_MODEL_PATH) 
print("\n========== TRAINING FINISHED ==========") 
print("Running one final validation on SHDB-AF...") 
val_loss,val_acc,val_precision,val_recall,val_f1,val_auc,val_time=validate(model,val_loader,criterion,DEVICE) 
print("\n========== FINAL VALIDATION ==========") 
print(f"VAL loss      : {val_loss:.6f}") 
print(f"VAL accuracy  : {val_acc:.4f}") 
print(f"VAL precision : {val_precision:.4f}") 
print(f"VAL recall    : {val_recall:.4f}") 
print(f"VAL F1        : {val_f1:.4f}") 
print(f"VAL AUC       : {val_auc:.4f}") 
print(f"VAL time      : {val_time/60:.1f} min") 
final_checkpoint={ 
    "epoch":EPOCHS, 
    "model_state_dict":model.state_dict(), 
    "optimizer_state_dict":optimizer.state_dict(), 
    "val_loss":val_loss, 
    "val_accuracy":val_acc, 
    "val_precision":val_precision, 
    "val_recall":val_recall, 
    "val_f1":val_f1, 
    "val_auc":val_auc, 
    "model_parameters":sum(p.numel() for p in model.parameters()), 
    "input_fs":125, 
    "window_sec":10, 
    "window_samples":1250, 
    "classes":["non-AF/AFL","AF/AFL"] 
} 
torch.save(final_checkpoint,FINAL_MODEL_PATH) 
print("\nFINAL MODEL SAVED:") 
print(FINAL_MODEL_PATH) 
