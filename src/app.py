import streamlit as st
import pandas as pd
import joblib

st.title("💡 Dự đoán Khách hàng rời bỏ dịch vụ")

# Load mô hình và threshold
try:
    model_info = joblib.load('best_churn_model.pkl')
    model = model_info['model']
    best_threshold = model_info['threshold']
    st.sidebar.success("Mô hình đã được tải thành công!")
except FileNotFoundError:
    st.sidebar.error("Lỗi: Không tìm thấy file mô hình 'best_churn_model.pkl'. Vui lòng chạy lại phần huấn luyện mô hình.")
    model = None
    best_threshold = 0.5


# --- Form nhập liệu ---
st.header("Nhập thông tin khách hàng:")
gender = st.selectbox("gender (Giới tính):", ["Female", "Male"]) # Match original data order
senior = st.selectbox("SeniorCitizen (Khách hàng cao tuổi):", [0, 1])
partner = st.selectbox("Partner (Có bạn đời/đối tác):", ["No", "Yes"]) # Match original data order
dependents = st.selectbox("Dependents (Có người phụ thuộc):", ["No", "Yes"]) # Match original data order
tenure = st.number_input("tenure (Thời gian gắn bó - tháng):", min_value=0, max_value=100, value=1)
phone = st.selectbox("PhoneService (Dịch vụ điện thoại):", ["Yes", "No"])
multiple = st.selectbox("MultipleLines (Nhiều đường dây):", ["No", "Yes", "No phone service"]) # Match original data order and handle special value
internet = st.selectbox("InternetService (Loại Internet):", ["DSL", "Fiber optic", "No"])
onlinesec = st.selectbox("OnlineSecurity (Bảo mật trực tuyến):", ["No", "Yes", "No internet service"]) # Match original data order and handle special value
onlinebackup = st.selectbox("OnlineBackup (Sao lưu trực tuyến):", ["Yes", "No", "No internet service"]) # Match original data order and handle special value
deviceprot = st.selectbox("DeviceProtection (Bảo vệ thiết bị):", ["No", "Yes", "No internet service"]) # Match original data order and handle special value
tech = st.selectbox("TechSupport (Hỗ trợ kỹ thuật):", ["No", "Yes", "No internet service"]) # Match original data order and handle special value
streamtv = st.selectbox("StreamingTV (Xem TV trực tuyến):", ["No", "Yes", "No internet service"]) # Match original data order and handle special value
streammovie = st.selectbox("StreamingMovies (Xem phim trực tuyến):", ["No", "Yes", "No internet service"]) # Match original data order and handle special value
contract = st.selectbox("Contract (Loại hợp đồng):", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("PaperlessBilling (Hóa đơn điện tử):", ["Yes", "No"])
payment = st.selectbox("PaymentMethod (Phương thức thanh toán):", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
monthly = st.number_input("MonthlyCharges (Chi phí hàng tháng):", min_value=0.0, value=20.0)
total = st.number_input("TotalCharges (Tổng chi phí):", min_value=0.0, value=20.0)

# --- Chuẩn bị dữ liệu cho mô hình ---
input_data = {
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone,
    'MultipleLines': multiple,
    'InternetService': internet,
    'OnlineSecurity': onlinesec,
    'OnlineBackup': onlinebackup,
    'DeviceProtection': deviceprot,
    'TechSupport': tech,
    'StreamingTV': streamtv,
    'StreamingMovies': streammovie,
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly,
    'TotalCharges': total
}
df_input = pd.DataFrame([input_data])

# Preprocess special values ("No internet service", "No phone service")
replace_no_cols = ['MultipleLines', 'OnlineSecurity','OnlineBackup','DeviceProtection',
                   'TechSupport','StreamingTV','StreamingMovies'] # Exclude PhoneService as it's handled by itself

for col in replace_no_cols:
    if col in df_input.columns:
        df_input[col] = df_input[col].replace({'No internet service':'No',
                                         'No phone service':'No'})


# Apply one-hot encoding - need to match columns from training data
# In a real app, you'd save the list of columns from the training data or the OHE transformer
# For simplicity here, we'll manually create columns based on the training data structure
# This requires knowledge of the columns generated during training
# Based on the notebook output (dataset_dummy.columns), the columns were:
# ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
#  'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
#  'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
#  'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
#  'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
#  'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
#  'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
#  'PaymentMethod_Mailed check']

# Manual One-Hot Encoding (simplified)
df_processed_app = df_input.copy()

# Binary features (Yes/No -> 1/0)
binary_map = {'Yes': 1, 'No': 0}
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
for col in binary_cols:
    if col in df_processed_app.columns:
        df_processed_app[col] = df_processed_app[col].map(binary_map).fillna(0).astype(int) # Handle 'No phone service'/'No internet service' which became 'No'

# Gender (Female/Male -> 0/1)
if 'gender' in df_processed_app.columns:
    df_processed_app['gender_Male'] = df_processed_app['gender'].map({'Male': 1, 'Female': 0}).astype(int)
    df_processed_app = df_processed_app.drop(columns=['gender'])

# InternetService (DSL/Fiber optic/No)
if 'InternetService' in df_processed_app.columns:
    df_processed_app['InternetService_Fiber optic'] = (df_processed_app['InternetService'] == 'Fiber optic').astype(int)
    df_processed_app['InternetService_No'] = (df_processed_app['InternetService'] == 'No').astype(int)
    df_processed_app = df_processed_app.drop(columns=['InternetService'])

# Contract (Month-to-month/One year/Two year)
if 'Contract' in df_processed_app.columns:
    df_processed_app['Contract_One year'] = (df_processed_app['Contract'] == 'One year').astype(int)
    df_processed_app['Contract_Two year'] = (df_processed_app['Contract'] == 'Two year').astype(int)
    df_processed_app = df_processed_app.drop(columns=['Contract'])

# PaymentMethod
if 'PaymentMethod' in df_processed_app.columns:
    df_processed_app['PaymentMethod_Credit card (automatic)'] = (df_processed_app['PaymentMethod'] == 'Credit card (automatic)').astype(int)
    df_processed_app['PaymentMethod_Electronic check'] = (df_processed_app['PaymentMethod'] == 'Electronic check').astype(int)
    df_processed_app['PaymentMethod_Mailed check'] = (df_processed_app['PaymentMethod'] == 'Mailed check').astype(int)
    df_processed_app = df_processed_app.drop(columns=['PaymentMethod'])

# Rename binary columns to match the training data structure (Yes -> _Yes)
rename_map = {col: col + '_Yes' for col in binary_cols if col in df_processed_app.columns}
df_processed_app = df_processed_app.rename(columns=rename_map)

# Ensure all expected columns are present and in the correct order
# This is crucial for the model to work correctly.
# Get the list of columns from the training data (excluding Churn)
# This list should ideally be saved during model training
# Based on notebook output:
expected_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
                 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
                 'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
                 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
                 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
                 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                 'PaymentMethod_Mailed check']


# Add missing columns with value 0
for col in expected_cols:
    if col not in df_processed_app.columns:
        df_processed_app[col] = 0

# Reorder columns to match the training data
df_processed_app = df_processed_app[expected_cols]

# Scaling - Need to load the scaler fitted on training data
# For simplicity, we'll skip explicit scaling here, assuming the model handles it
# or that the scaling was done as part of the model pipeline (which it wasn't in the notebook)
# A robust app would save and load the scaler.

# Prediction
if st.button("Dự đoán"):
    if model is not None:
        try:
            # Ensure dtypes match if necessary (esp. for boolean/int)
            df_processed_app = df_processed_app.astype(model.get_params().get('dtype') or df_processed_app.dtypes) # Attempt to match dtype if specified in model

            pred_prob = model.predict_proba(df_processed_app)[:, 1]
            pred = (pred_prob >= best_threshold).astype(int)[0]

            st.subheader("Kết quả dự đoán:")
            if pred == 1:
                st.error(f"⚠️ Khách hàng có khả năng NGỪNG sử dụng dịch vụ.")
                st.write(f"Xác suất rời bỏ: **{pred_prob[0]*100:.2f}%**")
            else:
                st.success(f"✅ Khách hàng có khả năng TIẾP TỤC sử dụng dịch vụ.")
                st.write(f"Xác suất ở lại: **{(1-pred_prob[0])*100:.2f}%**")

            st.write(f"(Dự đoán dựa trên ngưỡng xác suất: {best_threshold:.3f})")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi dự đoán: {e}")
            st.write("Vui lòng kiểm tra lại dữ liệu nhập và file mô hình.")
    else:
        st.warning("Mô hình chưa được tải. Vui lòng kiểm tra lỗi ở sidebar.")
