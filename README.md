# Dự án Dự đoán Khách hàng Rời bỏ Dịch vụ (Customer Churn Prediction)

Đây là một dự án mẫu về phân tích và xây dựng mô hình dự đoán khách hàng có khả năng rời bỏ dịch vụ viễn thông hay không, dựa trên bộ dữ liệu **Telco Customer Churn** của IBM.  
Dự án được tổ chức thành các module riêng biệt để dễ quản lý và phát triển.

---

## 🚀 Cài đặt và chạy dự án

```bash
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

---

## 📁 Cấu trúc thư mục và chức năng các file

```bash
.
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # File dữ liệu gốc
├── requirements.txt                    # Danh sách các thư viện cần cài đặt
├── preprocessing.py                    # Xử lý sơ bộ dữ liệu
├── eda.py                              # Khám phá dữ liệu & Trực quan hóa
├── feature_engineering.py              # Xử lý đặc trưng
├── model_training.py                   # Huấn luyện mô hình & Lưu mô hình
├── evaluation.py                       # Đánh giá mô hình
├── best_churn_model.pkl                # File lưu mô hình tốt nhất đã huấn luyện
└── app.py                              # Ứng dụng web Streamlit
```

### 🧩 Chức năng của từng file:

- **WA_Fn-UseC_-Telco-Customer-Churn.csv**: Bộ dữ liệu gốc về khách hàng.
- **requirements.txt**: Danh sách thư viện cần thiết.
- **preprocessing.py**: Xử lý dữ liệu ban đầu (missing values, kiểu dữ liệu, chuẩn hóa).
- **eda.py**: Phân tích và trực quan hóa dữ liệu.
- **feature_engineering.py**: Biến đổi đặc trưng, one-hot encoding, scaling.
- **model_training.py**: Huấn luyện Logistic Regression và XGBoost, lưu mô hình tốt nhất.
- **evaluation.py**: Đánh giá mô hình, tạo classification report, confusion matrix, threshold tối ưu.
- **best_churn_model.pkl**: Mô hình đã huấn luyện.
- **app.py**: Ứng dụng web Streamlit cho phép nhập thông tin khách hàng và dự đoán khả năng rời bỏ dịch vụ.

---

## 🧠 Tổng kết (Summary)

**Data Analysis Key Findings:**  
- Cấu trúc dự án được tổ chức rõ ràng theo module.  
- Đã tạo file `requirements.txt` chứa các thư viện: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `streamlit`, `joblib`.  
- `preprocessing.py` xử lý dữ liệu, chuyển đổi kiểu và chuẩn hóa.  
- `eda.py` chứa các biểu đồ phân tích dữ liệu churn.  
- `feature_engineering.py` xử lý encoding và scaling.  
- `model_training.py` huấn luyện Logistic Regression và XGBoost, áp dụng SMOTE, lưu mô hình tốt nhất.  
- `evaluation.py` đánh giá mô hình, tính F1-score, chọn threshold tối ưu.  
- `app.py` triển khai giao diện dự đoán bằng Streamlit.

**Next Steps:**  
- Cải thiện `app.py` bằng cách load transformers fitted từ tập huấn luyện để đảm bảo pipeline nhất quán.  
- Tạo thêm `utils.py` để tái sử dụng các hàm chung (load/save model, transform dữ liệu).

---

## 💻 Hướng dẫn chạy ứng dụng web Streamlit

1. **Cài đặt các thư viện cần thiết:**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Đảm bảo file mô hình tồn tại:**  
   File `best_churn_model.pkl` cần nằm cùng thư mục với `app.py`.

3. **Chạy ứng dụng Streamlit:**  
   ```bash
   streamlit run app.py
   ```

4. **Truy cập ứng dụng:**  
   - Local URL: http://localhost:8501  
   - Network URL: hiển thị trong terminal  
   - Nếu dùng ngrok (trên Colab), sẽ có địa chỉ public URL

5. **Dừng ứng dụng:**  
   Nhấn **Ctrl + C** trong terminal để dừng Streamlit.

---

© 2025 - Customer Churn Prediction Project
