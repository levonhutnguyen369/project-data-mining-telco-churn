import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler # Giả định MinMaxScaler có thể được sử dụng trong feature engineering

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Đánh giá mô hình đã huấn luyện và in báo cáo phân loại.

    Args:
        model: Mô hình đã huấn luyện.
        X_test: Đặc trưng kiểm tra.
        y_test: Biến mục tiêu kiểm tra.
        threshold: Ngưỡng xác suất để phân loại.
    """
    print(f"\n--- Báo cáo Đánh giá (Ngưỡng = {threshold:.3f}) ---")
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype(int)
    else:
         y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

def plot_confusion_matrix(y_true, y_pred, title="Ma trận Nhầm lẫn"):
    """
    Vẽ biểu đồ ma trận nhầm lẫn dưới dạng heatmap.

    Args:
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        title: Tiêu đề biểu đồ.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title(title)
    plt.show()

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

    # Đảm bảo mảng thresholds không rỗng trước khi tính toán F1 scores
    if len(thresholds) == 0:
        print("Không tìm thấy ngưỡng cho đường cong precision-recall.")
        return 0.5

    # Thêm 0.5 vào thresholds nếu chưa có để đánh giá mặc định
    if 0.5 not in thresholds:
         thresholds = np.append(thresholds, 0.5)
         # Tính toán lại precisions và recalls cho mảng thresholds mới
         # Đây là một cách đơn giản hóa; phương pháp đúng hơn là nội suy hoặc đánh giá lại
         # Để đơn giản ở đây, chúng ta chỉ đảm bảo 0.5 được xem xét.
         # Một phương pháp mạnh mẽ hơn sẽ đánh giá tại các ngưỡng cụ thể.
         # Hiện tại, chúng ta dựa vào precision_recall_curve cung cấp một phạm vi tốt.


    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)

    if len(f1_scores) == 0:
         print("Không thể tính F1 scores.")
         return 0.5

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    print(f"✅ Tìm thấy ngưỡng tốt nhất: {best_thresh:.3f} (F1-score: {f1_scores[best_idx]:.3f})")
    return best_thresh


def perform_cross_validation(model, X, y, cv=5, scoring='f1'):
    """
    Thực hiện xác thực chéo và in kết quả.

    Args:
        model: Mô hình cần đánh giá.
        X: Đặc trưng.
        y: Biến mục tiêu.
        cv: Số fold cho xác thực chéo.
        scoring: Chỉ số đánh giá (ví dụ: 'f1', 'accuracy', 'roc_auc').
    """
    print(f"\n--- Thực hiện Xác thực chéo {cv}-fold ({scoring}) ---")
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"{scoring}-score từng fold: {scores}")
    print(f"Trung bình {scoring}-score: {scores.mean():.3f}")


if __name__ == '__main__':
    # Khối này minh họa cách sử dụng các hàm đánh giá
    # Trong một ứng dụng thực tế, bạn sẽ load, tiền xử lý, kỹ thuật đặc trưng,
    # và chia dữ liệu trước khi sử dụng các hàm này.

    # Để minh họa, chúng ta sẽ tạo dữ liệu giả hoặc giả định nó có sẵn
    try:
        print("Sử dụng dữ liệu train/test có sẵn từ môi trường notebook để minh họa đánh giá.")

        # Đảm bảo X và y có sẵn cho minh họa xác thực chéo
        if 'X' not in globals() or 'y' not in globals():
             print("Không tìm thấy X hoặc y. Đang tạo dữ liệu giả cho minh họa xác thực chéo.")
             # Tạo dữ liệu giả nếu không có sẵn
             from sklearn.datasets import make_classification
             X, y = make_classification(n_samples=1000, n_features=20, n_classes=2,
                                       weights=[0.8, 0.2], random_state=42)
             X = pd.DataFrame(X) # Chuyển thành DataFrame cho nhất quán
             y = pd.Series(y)    # Chuyển thành Series cho nhất quán


        # Đảm bảo X_train, X_test, y_train, y_test có sẵn cho minh họa đánh giá
        if 'X_train' not in globals() or 'X_test' not in globals() or \
           'y_train' not in globals() or 'y_test' not in globals():
            print("Không tìm thấy dữ liệu chia train/test. Đang thực hiện chia dữ liệu cho minh họa đánh giá.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        # Tạo mô hình giả hoặc sử dụng mô hình đã huấn luyện trong model_training.py
        # Giả định các mô hình đã huấn luyện có thể có sẵn từ trạng thái notebook
        if 'log_model_no_smote' in globals():
            print("\nĐang đánh giá Logistic Regression (Không SMOTE):")
            best_thresh_log = find_best_threshold(log_model_no_smote, X_test, y_test)
            evaluate_model(log_model_no_smote, X_test, y_test, threshold=best_thresh_log)
            plot_confusion_matrix(y_test, (log_model_no_smote.predict_proba(X_test)[:, 1] >= best_thresh_log).astype(int), "Ma trận Nhầm lẫn - Logistic Regression")
            perform_cross_validation(log_model_no_smote, X, y)

        if 'xgb_model_no_smote' in globals():
            print("\nĐang đánh giá XGBoost (Không SMOTE):")
            best_thresh_xgb = find_best_threshold(xgb_model_no_smote, X_test, y_test)
            evaluate_model(xgb_model_no_smote, X_test, y_test, threshold=best_thresh_xgb)
            plot_confusion_matrix(y_test, (xgb_model_no_smote.predict_proba(X_test)[:, 1] >= best_thresh_xgb).astype(int), "Ma trận Nhầm lẫn - XGBoost")
            perform_cross_validation(xgb_model_no_smote, X, y)

        # Ví dụ nếu mô hình được huấn luyện với SMOTE và lưu lại
        # try:
        #     import joblib
        #     best_model_info = joblib.load('best_churn_model.pkl')
        #     loaded_model = best_model_info['model']
        #     loaded_threshold = best_model_info['threshold']
        #     print(f"\nĐang đánh giá mô hình đã load (XGBoost với SMOTE) sử dụng ngưỡng {loaded_threshold:.3f}:")
        #     evaluate_model(loaded_model, X_test, y_test, threshold=loaded_threshold)
        #     plot_confusion_matrix(y_test, (loaded_model.predict_proba(X_test)[:, 1] >= loaded_threshold).astype(int), "Ma trận Nhầm lẫn - XGBoost Đã Load")
        #     # Lưu ý: Cross-validation với SMOTE yêu cầu một pipeline, không thực hiện ở đây để đơn giản
        # except FileNotFoundError:
        #     print("\nKhông tìm thấy 'best_churn_model.pkl'. Bỏ qua đánh giá mô hình đã load.")


    except NameError:
        print("\nKhông tìm thấy dữ liệu hoặc mô hình cần thiết trong môi trường.")
        print("Để chạy script này độc lập, vui lòng đảm bảo đã thực hiện load dữ liệu, tiền xử lý, kỹ thuật đặc trưng và huấn luyện mô hình trước.")
