26/09/2026
Plan train model AI hiện tại để phát hiện rung nhĩ AF
Giai đoạn hiện tại ===> Train model AI sử dụng 1D ConvNextV2 trên các tập Dataset chuẩn và test trên các tập Dataset chuẩn khác
https://www.mdpi.com/1424-8220/25/13/4109 ( link bài báo Atrial Fibrillation and Atrial Flutter Detection Using Deep Learning
by Dimitri Kraft , andPeter Rumm )

Em tham khảo cách triển khai 1D ConvNeXtV2 từ repository cfauchereau/ecg-convnext, đặc biệt là cấu trúc Block gồm Depthwise Conv1D, LayerNorm, pointwise expansion ×4, GELU, GRN và residual connection. ==> link github https://github.com/cfauchereau/ecg-convnext/blob/main/ecg_convnext.py

Link code  GG colab em đang sử dụng:  https://colab.research.google.com/drive/1v9Sj8JcebEM4shU1_jjh4y0maMgZGHby?usp=sharing

Mô hình hiện tại bao gồm:
- Training dataset: PTB-XL
- Validating dataset: SHDB-AF
- Testing dataset: MIT-AFDB
  
  ==> Mục tiêu làm mô hình: để làm benchmark (F1-score,AUC,và Accuracy) khi test trên các tập dữ liệu chuẩn khác, VD là Icentia11k
  
  ==> Kết quả hiện tại: kết quả từ test cho thấy model dự đoán toàn bộ MIT-AFDB là non-AF nên Recall/F1 bằng 0, nguyên nhân chính là tập train PTB-XL quá mất cân       bằng (AF chỉ khoảng 7,2%) nên model bị lệch về class 0.
  
  ==> Nhiệm vụ tự đặt ra trong tuần tới: fix lỗi imbalance và đọc thêm nhiều báo để phát triển thêm hướng tư duy 

Em đã tiền xử lý 3 bộ dữ liệu PTB-XL, SHDB-AF và MIT-AFDB về cùng tần số lấy mẫu 125 Hz, dùng tín hiệu ECG I-lead và chia thành các cửa sổ 10 giây tương ứng 1250 mẫu 
==>Lý do chuẩn hóa về 125Hz là để tất cả dữ liệu có cùng đầu vào cho mô hình, đồng thời bám sát thiết lập của bài báo, trong đó tác giả cũng resample ECG về 125 Hz để giảm chi phí tính toán và lưu trữ nhưng vẫn giữ đủ thông tin cho phân tích P-wave và QRS.

Khi chạy trên gg colab, em chia thành 3 cell:
    
    Cell 1: đọc 3 file HDF5, tạo Dataset/DataLoader, lọc nhãn không hợp lệ của SHDB-AF, và xây dựng mô hình 1D ConvNeXtV2 theo kiến trúc bài báo với 4 stage 16–32–64–128 và số block 3–3–9–3.     
    Cell 2: huấn luyện mô hình trên PTB-XL trong 20 epoch, áp dụng augmentation theo bài báo, sau đó validation trên SHDB-AF và lưu trọng số model đã train vào Google Drive.     
    Cell 3: load model đã train, test trên MIT-AFDB, tính các chỉ số Accuracy, Precision, Recall/Sensitivity, F1, ROC-AUC, confusion matrix, sau đó đánh giá thêm theo cách bài báo với threshold 0.75, merge gap 40 s và duration-based Sensitivity/Precision/F1.


Đây là link gg Drive ạ: https://drive.google.com/drive/folders/18WN2C4afJ8TluNBDwLykx73f0dc2asVr?usp=sharing
Trong đó các file quan trọng bao gồm: 

- preprocessed: chứa dữ liệu ECG đã tiền xử lý của 3 dataset,  định dạng  125 Hz, cửa sổ 10 s, HDF5 và nhãn tương ứng.
- final_model_epoch20.pth: file model 1D ConvNeXtV2 sau khi train 20 epoch, có trọng số model và một số thông tin checkpoint như optimizer, epoch, metric validation.
- mit_afdb_test_results.npz: chứa kết quả test trên MIT-AFDB, như y_true, xác suất p_AF, dự đoán, Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix.
-  dataset: là dữ liệu gốc ban đầu, chưa qua tiền xử lý
