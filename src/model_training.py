import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler cho nhất quán với FE

def train_logistic_regression(X_train, y_train, use_smote=False):
    """
    Huấn luyện mô hình Logistic Regression.

    Args:
        X_train: Đặc trưng huấn luyện.
        y_train: Biến mục tiêu huấn luyện.
        use_smote: Có áp dụng SMOTE cho dữ liệu huấn luyện hay không.

    Returns:
        Mô hình LogisticRegression đã huấn luyện.
    """
    print("Đang huấn luyện mô hình Logistic Regression...")
    if use_smote:
        print("Áp dụng SMOTE...")
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print(f"Kích thước dữ liệu huấn luyện sau SMOTE: {X_train_res.shape}")
    else:
        X_train_res, y_train_res = X_train, y_train

    # Sử dụng các tham số tìm được trong quá trình tuning ở notebook
    model = LogisticRegression(C=1, penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
    model.fit(X_train_res, y_train_res)
    return model

def train_xgboost(X_train, y_train, use_smote=False):
    """
    Huấn luyện mô hình XGBoost.

    Args:
        X_train: Đặc trưng huấn luyện.
        y_train: Biến mục tiêu huấn luyện.
        use_smote: Có áp dụng SMOTE cho dữ liệu huấn luyện hay không.

    Returns:
        Mô hình XGBoostClassifier đã huấn luyện.
    """
    print("Đang huấn luyện mô hình XGBoost...")
    if use_smote:
        print("Áp dụng SMOTE...")
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print(f"Kích thước dữ liệu huấn luyện sau SMOTE: {X_train_res.shape}")
    else:
        X_train_res, y_train_res = X_train, y_train

    # Sử dụng các tham số tìm được trong quá trình tuning ở notebook
    model = XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=300,
                          subsample=0.8, colsample_bytree=0.8,
                          use_label_encoder=False, eval_metric='logloss',
                          random_state=42)
    model.fit(X_train_res, y_train_res)
    return model

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Đánh giá mô hình đã huấn luyện và in báo cáo phân loại.

    Args:
        model: Mô hình đã huấn luyện.
        X_test: Đặc trưng kiểm tra.
        y_test: Biến mục tiêu kiểm tra.
        threshold: Ngưỡng xác suất để phân loại.
    """
    print(f"\nĐang đánh giá mô hình với ngưỡng = {threshold}")
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype(int)
    else:
         y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    # Tuỳ chọn: Vẽ confusion matrix nếu cần, nhưng yêu cầu thiết lập matplotlib/seaborn

def find_best_threshold(model, X_test, y_test):
    """
    Tìm ngưỡng xác suất tốt nhất dựa trên F1-score.

    Args:
        model: Mô hình đã huấn luyện có phương thức predict_proba.
        X_test: Đặc trưng kiểm tra.
        y_test: Biến mục tiêu kiểm tra.

    Returns:
        Ngưỡng tốt nhất tìm được.
    """
    if not hasattr(model, 'predict_proba'):
        print("Mô hình không hỗ trợ predict_proba. Không thể tìm ngưỡng tốt nhất.")
        return 0.5 # Trả về mặc định

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    print(f"✅ Tìm thấy ngưỡng tốt nhất: {best_thresh:.3f}  |  Best F1-score: {f1_scores[best_idx]:.3f}")
    return best_thresh


if __name__ == '__main__':
    # Khối này minh họa cách sử dụng các hàm
    # Trong một ứng dụng thực tế, bạn sẽ load và tiền xử lý dữ liệu trước.
    # Ví dụ:
    # from preprocessing import preprocess_inputs
    # from feature_engineering import perform_feature_engineering
    #
    # df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # df_processed = preprocess_inputs(df)
    # X, y = perform_feature_engineering(df_processed)
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )

    # Để minh họa, giả định X_train, y_train, X_test, y_test
    # có sẵn từ trạng thái môi trường notebook.
    try:
        print("Sử dụng dữ liệu train/test có sẵn từ môi trường notebook.")

        # Huấn luyện Logistic Regression (với SMOTE vì nó hoạt động tốt hơn trong tuning)
        log_model_smote = train_logistic_regression(X_train, y_train, use_smote=True)
        print("\n--- Đánh giá Logistic Regression (SMOTE) ---")
        # Tìm và đánh giá với ngưỡng tốt nhất
        best_thresh_log = find_best_threshold(log_model_smote, X_test, y_test)
        evaluate_model(log_model_smote, X_test, y_test, threshold=best_thresh_log)


        # Huấn luyện XGBoost (với SMOTE)
        xgb_model_smote = train_xgboost(X_train, y_train, use_smote=True)
        print("\n--- Đánh giá XGBoost (SMOTE) ---")
        # Tìm và đánh giá với ngưỡng tốt nhất
        best_thresh_xgb = find_best_threshold(xgb_model_smote, X_test, y_test)
        evaluate_model(xgb_model_smote, X_test, y_test, threshold=best_thresh_xgb)


        # Huấn luyện Logistic Regression (không SMOTE)
        log_model_no_smote = train_logistic_regression(X_train, y_train, use_smote=False)
        print("\n--- Đánh giá Logistic Regression (Không SMOTE) ---")
        best_thresh_log_no_smote = find_best_threshold(log_model_no_smote, X_test, y_test)
        evaluate_model(log_model_no_smote, X_test, y_test, threshold=best_thresh_log_no_smote)


        # Huấn luyện XGBoost (không SMOTE)
        xgb_model_no_smote = train_xgboost(X_train, y_train, use_smote=False)
        print("\n--- Đánh giá XGBoost (Không SMOTE) ---")
        best_thresh_xgb_no_smote = find_best_threshold(xgb_model_no_smote, X_test, y_test)
        evaluate_model(xgb_model_no_smote, X_test, y_test, threshold=best_thresh_xgb_no_smote)


        # Quyết định mô hình nào là "tốt nhất" dựa trên một chỉ số được chọn (ví dụ: F1-score hoặc recall trên churn=1)
        # Để lưu lại, chúng ta sẽ lưu mô hình XGBoost được huấn luyện với SMOTE và ngưỡng tốt nhất của nó,
        # vì XGBoost thường hoạt động tốt, và SMOTE giúp xử lý dữ liệu mất cân bằng.
        # Bạn có thể chọn một mô hình/ngưỡng khác tùy thuộc vào mục tiêu kinh doanh của bạn.
        print("\nĐang lưu mô hình hoạt động tốt nhất (XGBoost với SMOTE và ngưỡng tốt nhất)...")
        best_model_to_save = {
            'model': xgb_model_smote,
            'threshold': best_thresh_xgb
        }
        joblib.dump(best_model_to_save, 'best_churn_model.pkl')
        print("Mô hình đã được lưu dưới tên 'best_churn_model.pkl'")


    except NameError:
        print("\nKhông tìm thấy X_train, y_train, X_test, y_test.")
        print("Để chạy script này độc lập, vui lòng đảm bảo đã thực hiện load dữ liệu, tiền xử lý và chia train/test trước.")
