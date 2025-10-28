import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def perform_feature_engineering(df):
    """
    Thực hiện kỹ thuật đặc trưng bao gồm one-hot encoding và scaling.

    Args:
        df: pandas DataFrame với dữ liệu đã được xử lý (sau khi chạy preprocessing.py).

    Returns:
        Tuple chứa:
            - X: pandas DataFrame với các đặc trưng đã được xử lý.
            - y: pandas Series với biến mục tiêu (Churn).
    """
    data = df.copy()

    # Tách đặc trưng và biến mục tiêu
    if 'Churn' in data.columns:
        X = data.drop("Churn", axis=1)
        y = data["Churn"]
    else:
        X = data # Giả định df đầu vào đã chỉ chứa các đặc trưng
        y = None # Biến mục tiêu không có sẵn

    # Xác định các đặc trưng phân loại và số
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Áp dụng one-hot encoding cho các đặc trưng phân loại
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Áp dụng scaling cho các đặc trưng số (sử dụng MinMaxScaler như trong notebook)
    # Trong môi trường thực tế, scaler chỉ nên fit trên dữ liệu huấn luyện,
    # nhưng để đơn giản trong hàm này, chúng ta fit trên toàn bộ X
    scaler = MinMaxScaler()
    if numeric_features: # Kiểm tra xem có đặc trưng số nào không
        X[numeric_features] = scaler.fit_transform(X[numeric_features])


    return X, y

if __name__ == '__main__':
    # Ví dụ sử dụng (nếu chạy trực tiếp file này)
    # Giả định bạn có một DataFrame mẫu hoặc load từ file
    # Để minh họa, sử dụng cấu trúc dữ liệu đã tiền xử lý
    # Trong một script độc lập, bạn sẽ load và tiền xử lý dữ liệu trước
    try:
        # Giả định df_processed có sẵn từ môi trường notebook
        print("Đang thực hiện kỹ thuật đặc trưng trên df_processed có sẵn...")
        X_engineered, y_target = perform_feature_engineering(df_processed.copy())
        print("\nEngineered Features (X) head:")
        print(X_engineered.head())
        print("\nTarget (y) head:")
        print(y_target.head())
        print("\nEngineered Features shape:", X_engineered.shape)
        print("Target shape:", y_target.shape)

    except NameError:
        print("Không tìm thấy df_processed. Vui lòng đảm bảo dữ liệu đã được load và tiền xử lý.")
        print("Để chạy script này độc lập, load và tiền xử lý dữ liệu trước khi gọi perform_feature_engineering.")
