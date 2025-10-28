import pandas as pd
import numpy as np

def preprocess_inputs(df):
  """
  Tiền xử lý dữ liệu thô về khách hàng rời bỏ dịch vụ.

  Args:
    df: pandas DataFrame chứa dữ liệu thô.

  Returns:
    pandas DataFrame với dữ liệu đã được xử lý.
  """
  data = df.copy()

  # Xóa cột không cần thiết
  if 'customerID' in df.columns:
    data = data.drop(columns=['customerID'])

  # Xử lý Total Charges rỗng và chuyển sang kiểu số
  data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

  if data["TotalCharges"].isnull().sum() > 0:
    data = data.dropna(subset=['TotalCharges']).reset_index(drop=True)

  # Chuẩn whitespace cho các cột object
  for c in data.select_dtypes(include='object').columns:
    data[c] = data[c].str.strip()

  # Chuyển biến mục tiêu Churn -> 0/1 và đảm bảo kiểu số nguyên
  data['Churn'] = data['Churn'].astype(str).map({'Yes': 1, 'No': 0}).astype('int64')

  # Chuẩn hóa các giá trị đặc biệt ("No internet service", "No phone service")
  replace_no = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                'TechSupport','StreamingTV','StreamingMovies','MultipleLines']

  for col in replace_no:
      if col in data.columns:
          data[col] = data[col].replace({'No internet service':'No',
                                           'No phone service':'No'})

  return data

if __name__ == '__main__':
    # Ví dụ sử dụng (nếu chạy trực tiếp file này)
    # Giả định bạn có một DataFrame mẫu hoặc load từ file
    # Để minh họa, tạo một DataFrame mẫu đơn giản
    dummy_data = {
        'customerID': ['1', '2', '3'],
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes'],
        'tenure': [12, 24, 36],
        'PhoneService': ['Yes', 'No', 'Yes'],
        'MultipleLines': ['No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'No', 'Fiber optic'],
        'OnlineSecurity': ['No', 'No internet service', 'Yes'],
        'OnlineBackup': ['Yes', 'No internet service', 'No'],
        'DeviceProtection': ['No', 'No internet service', 'Yes'],
        'TechSupport': ['No', 'No internet service', 'Yes'],
        'StreamingTV': ['No', 'No internet service', 'Yes'],
        'StreamingMovies': ['No', 'No internet service', 'Yes'],
        'Contract': ['One year', 'Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [50.0, 20.0, 100.0],
        'TotalCharges': [600.0, ' ', 3600.0],
        'Churn': ['No', 'Yes', 'No']
    }
    dummy_df = pd.DataFrame(dummy_data)
    processed_df = preprocess_inputs(dummy_df)
    print("Processed DataFrame:")
    print(processed_df)
