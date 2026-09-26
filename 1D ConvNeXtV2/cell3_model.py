import os,time,h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================
BEST_MODEL_PATH="/content/drive/MyDrive/AF_ConvNeXtV2/best_model.pth"
FINAL_MODEL_PATH="/content/drive/MyDrive/AF_ConvNeXtV2/final_model_epoch20.pth"
if os.path.exists(BEST_MODEL_PATH):
    MODEL_PATH=BEST_MODEL_PATH
elif os.path.exists(FINAL_MODEL_PATH):
    MODEL_PATH=FINAL_MODEL_PATH
else:
    raise FileNotFoundError("Không tìm thấy best_model.pth hoặc final_model_epoch20.pth")
print("========== CELL 3 ==========")
print("Device:",DEVICE)
print("Loading model:",MODEL_PATH)
checkpoint=torch.load(MODEL_PATH,map_location=DEVICE,weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model=model.to(DEVICE)
model.eval()
print("Model loaded successfully")
# ============================================================
# 2. RUN MIT-AFDB INFERENCE
# test_loader from CELL 1 has shuffle=False
# ============================================================
all_y=[]
all_prob=[]
t0=time.time()
print("\n========== MIT-AFDB INFERENCE ==========")
with torch.no_grad():
    for batch_idx,(x,y) in enumerate(test_loader,1):
        x=x.to(DEVICE,non_blocking=True)
        logits=model(x)
        prob_af=torch.softmax(logits,dim=1)[:,1]
        all_y.append(y.numpy())
        all_prob.append(prob_af.cpu().numpy())
        if batch_idx%100==0 or batch_idx==len(test_loader):
            print(f"\rTEST {batch_idx}/{len(test_loader)}",end="")
print()
y_true=np.concatenate(all_y).astype(np.uint8)
p_af=np.concatenate(all_prob).astype(np.float32)
test_time=time.time()-t0
print("Windows tested:",len(y_true))
print(f"Inference time: {test_time/60:.2f} min")
# ============================================================
# 3. WINDOW-LEVEL METRICS
# Standard binary threshold = 0.50
# ROC-AUC uses continuous p_AF
# ============================================================
y_pred_05=(p_af>=0.50).astype(np.uint8)
accuracy=accuracy_score(y_true,y_pred_05)
precision=precision_score(y_true,y_pred_05,zero_division=0)
recall=recall_score(y_true,y_pred_05,zero_division=0)
f1=f1_score(y_true,y_pred_05,zero_division=0)
auc=roc_auc_score(y_true,p_af)
cm=confusion_matrix(y_true,y_pred_05)
tn,fp,fn,tp=cm.ravel()
print("\n========== WINDOW-LEVEL RESULTS ==========")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall/Sensitivity : {recall:.4f}")
print(f"F1-score           : {f1:.4f}")
print(f"ROC-AUC            : {auc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print(f"TN={tn} | FP={fp} | FN={fn} | TP={tp}")
# ============================================================
# 4. LOAD MIT-AFDB TEMPORAL INFORMATION
# Same order as test_loader because shuffle=False
# ============================================================
with h5py.File(TEST_H5,"r") as f:
    records=np.asarray(f["record"][:])
    starts=np.asarray(f["start_sample_125hz"][:],dtype=np.int64)
if records.dtype.kind=="S":
    records=np.char.decode(records,"utf-8")
else:
    records=records.astype(str)
if len(records)!=len(y_true):
    raise RuntimeError("Metadata order/length does not match inference output")
# ============================================================
# 5. PAPER POST-PROCESSING
# p_AF >= 0.75
# merge AF regions separated by <= 40 s of non-AF
# Window = 10 s
# ============================================================
PAPER_THRESHOLD=0.75
WINDOW_SEC=10
MERGE_GAP_SEC=40
MERGE_GAP_WINDOWS=MERGE_GAP_SEC//WINDOW_SEC
y_pred_paper=(p_af>=PAPER_THRESHOLD).astype(np.uint8)
def merge_af_gaps(labels,max_gap_windows=4):
    labels=np.asarray(labels,dtype=np.uint8).copy()
    af_idx=np.where(labels==1)[0]
    if len(af_idx)<2:
        return labels
    for i in range(len(af_idx)-1):
        left=af_idx[i]
        right=af_idx[i+1]
        gap=right-left-1
        if 0<gap<=max_gap_windows:
            labels[left+1:right]=1
    return labels
# ============================================================
# 6. DURATION-BASED OVERLAP METRICS
# Evaluate record-by-record to avoid joining different patients
# ============================================================
total_ref_sec=0.0
total_pred_sec=0.0
total_overlap_sec=0.0
per_record=[]
unique_records=[]
for r in records:
    if r not in unique_records:
        unique_records.append(r)
for record in unique_records:
    idx=np.where(records==record)[0]
    order=np.argsort(starts[idx])
    idx=idx[order]
    ref=y_true[idx].astype(np.uint8)
    pred=y_pred_paper[idx].astype(np.uint8)
    pred_merged=merge_af_gaps(pred,MERGE_GAP_WINDOWS)
    ref_sec=float(ref.sum()*WINDOW_SEC)
    pred_sec=float(pred_merged.sum()*WINDOW_SEC)
    overlap_sec=float(np.logical_and(ref==1,pred_merged==1).sum()*WINDOW_SEC)
    total_ref_sec+=ref_sec
    total_pred_sec+=pred_sec
    total_overlap_sec+=overlap_sec
    rec_sens=overlap_sec/ref_sec if ref_sec>0 else np.nan
    rec_prec=overlap_sec/pred_sec if pred_sec>0 else np.nan
    rec_f1=2*rec_sens*rec_prec/(rec_sens+rec_prec) if np.isfinite(rec_sens) and np.isfinite(rec_prec) and (rec_sens+rec_prec)>0 else np.nan
    per_record.append((record,rec_sens,rec_prec,rec_f1,ref_sec,pred_sec,overlap_sec))
paper_sensitivity=total_overlap_sec/total_ref_sec if total_ref_sec>0 else np.nan
paper_precision=total_overlap_sec/total_pred_sec if total_pred_sec>0 else np.nan
paper_f1=2*paper_sensitivity*paper_precision/(paper_sensitivity+paper_precision) if (paper_sensitivity+paper_precision)>0 else np.nan
print("\n========== PAPER-STYLE MIT-AFDB RESULTS ==========")
print(f"AF probability threshold : {PAPER_THRESHOLD}")
print(f"Window size              : {WINDOW_SEC} s")
print(f"Merge gap                : {MERGE_GAP_SEC} s")
print(f"Reference AF duration    : {total_ref_sec/3600:.4f} h")
print(f"Predicted AF duration    : {total_pred_sec/3600:.4f} h")
print(f"Overlap AF duration      : {total_overlap_sec/3600:.4f} h")
print(f"Duration Sensitivity     : {paper_sensitivity:.4f}")
print(f"Duration Precision       : {paper_precision:.4f}")
print(f"Duration F1              : {paper_f1:.4f}")
# ============================================================
# 7. OPTIONAL: PER-RECORD PAPER-STYLE RESULTS
# ============================================================
print("\n========== PER-RECORD RESULTS ==========")
print(f"{'Record':<10}{'Sens':>10}{'Prec':>10}{'F1':>10}")
for record,sens,prec,f1_r,ref_sec,pred_sec,overlap_sec in per_record:
    sens_txt=f"{sens:.4f}" if np.isfinite(sens) else "N/A"
    prec_txt=f"{prec:.4f}" if np.isfinite(prec) else "N/A"
    f1_txt=f"{f1_r:.4f}" if np.isfinite(f1_r) else "N/A"
    print(f"{record:<10}{sens_txt:>10}{prec_txt:>10}{f1_txt:>10}")
# ============================================================
# 8. SAVE RESULTS TO DRIVE
# ============================================================
RESULT_PATH="/content/drive/MyDrive/AF_ConvNeXtV2/mit_afdb_test_results.npz"
np.savez(
    RESULT_PATH,
    y_true=y_true,
    p_af=p_af,
    y_pred_05=y_pred_05,
    y_pred_paper=y_pred_paper,
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1=f1,
    auc=auc,
    confusion_matrix=cm,
    paper_sensitivity=paper_sensitivity,
    paper_precision=paper_precision,
    paper_f1=paper_f1
)
print("\n========== CELL 3 FINISHED ==========")
print("Results saved:",RESULT_PATH)
