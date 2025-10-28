import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_distribution(df):
    """Vẽ biểu đồ phân bố của cột Churn."""
    plt.figure(figsize=(6,4))
    ax = sns.countplot(data=df, x='Churn')
    plt.title('Phân bố Churn (0=No, 1=Yes)')
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom')
    plt.show()

def plot_churn_by_gender(df):
    """Vẽ biểu đồ tỷ lệ Churn theo giới tính."""
    churn_by_gender = df.groupby('gender')['Churn'].value_counts(normalize=True).unstack().mul(100)
    ax = churn_by_gender.plot(kind='bar', stacked=False, figsize=(6, 4), color=['skyblue', 'salmon'])
    plt.title('Tỷ lệ Churn theo giới tính')
    plt.ylabel('Tỷ lệ (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Churn', labels=['No', 'Yes'])
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_churn_by_contract(df):
    """Vẽ biểu đồ tỷ lệ Churn theo loại hợp đồng."""
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x='Contract', hue='Churn')
    plt.title('Contract vs Churn')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.show()

def plot_churn_by_internet_service(df):
    """Vẽ biểu đồ tỷ lệ Churn theo loại dịch vụ internet."""
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x='InternetService', hue='Churn')
    plt.title('InternetService vs Churn')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.show()

def plot_tenure_distribution_by_churn(df):
    """Vẽ biểu đồ phân phối thời gian gắn bó (tenure) theo Churn."""
    plt.figure(figsize=(8,4))
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
    plt.title('Phân phối Tenure theo Churn')
    plt.show()

def plot_monthly_charges_distribution_by_churn(df):
    """Vẽ biểu đồ phân phối chi phí hàng tháng theo Churn."""
    plt.figure(figsize=(8,4))
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple='stack', bins=30)
    plt.title('Phân phối MonthlyCharges theo Churn')
    plt.show()

def plot_total_charges_distribution_by_churn(df):
    """Vẽ biểu đồ phân phối tổng chi phí theo Churn."""
    plt.figure(figsize=(8,4))
    sns.histplot(data=df, x='TotalCharges', hue='Churn', multiple='stack', bins=30)
    plt.title('Phân phối TotalCharges theo Churn')
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.show()

def plot_tenure_group_churn(df):
    """Vẽ biểu đồ tỷ lệ Churn theo nhóm thời gian gắn bó."""
    data_copy = df.copy()
    data_copy['tenure_group'] = pd.cut(data_copy['tenure'], bins=[0, 12, 24, 36, 48, 60, 72],
                                labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data_copy, x="tenure_group", hue="Churn", palette="Set2")
    plt.title("Tỉ lệ rời bỏ theo nhóm thời gian gắn bó (tenure)")
    plt.xlabel("Nhóm thời gian gắn bó (tháng)")
    plt.ylabel("Số lượng khách hàng")
    plt.show()

def plot_numeric_heatmap(df):
    """Vẽ heatmap ma trận tương quan cho các đặc trưng số."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(6,4))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Ma trận tương quan (numeric)')
    plt.show()

def plot_service_usage(df):
    """Tính toán và vẽ biểu đồ tỷ lệ sử dụng các dịch vụ."""
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    # Giả định 'Yes' nghĩa là sử dụng, các giá trị khác nghĩa là không sử dụng
    df_service = df[service_cols].applymap(lambda x: 1 if x == 'Yes' else 0)
    service_usage = round(df_service.mean().sort_values(ascending=False) * 100, 2)

    print("🔝 Top 3 dịch vụ được sử dụng nhiều nhất:")
    print(service_usage.head(3))
    print("\n🔻 Top 3 dịch vụ được sử dụng ít nhất:")
    print(service_usage.tail(3))

    plt.figure(figsize=(10,5))
    service_usage.sort_values(ascending=True).plot(kind='barh', color='skyblue')
    plt.title("Tỷ lệ khách hàng sử dụng từng dịch vụ (%)")
    plt.xlabel("Tỷ lệ sử dụng (Yes)")
    plt.ylabel("Dịch vụ")
    plt.show()


if __name__ == '__main__':
    # Ví dụ sử dụng: Load dữ liệu đã xử lý và chạy các hàm EDA
    # Trong môi trường thực tế, bạn sẽ load dữ liệu đã tiền xử lý ở đây.
    # Để minh họa, chúng ta sẽ sử dụng biến df_processed từ trạng thái notebook
    # Trong một script độc lập, bạn thường sẽ load dữ liệu:
    # from preprocessing import preprocess_inputs
    # df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # df_processed = preprocess_inputs(df)

    # Giả định df_processed có sẵn trong môi trường để minh họa
    # Nếu chạy như một script, bạn cần load/tiền xử lý dữ liệu
    try:
        # Phần này dùng để minh họa trong môi trường notebook
        # Trong một script độc lập, bạn sẽ load dữ liệu ở đây
        print("Đang chạy EDA với df_processed có sẵn...")
        plot_churn_distribution(df_processed)
        plot_churn_by_gender(df_processed)
        plot_churn_by_contract(df_processed)
        plot_churn_by_internet_service(df_processed)
        plot_tenure_distribution_by_churn(df_processed)
        plot_monthly_charges_distribution_by_churn(df_processed)
        plot_total_charges_distribution_by_churn(df_processed)
        plot_tenure_group_churn(df_processed)
        plot_numeric_heatmap(df_processed)
        plot_service_usage(df_processed)

    except NameError:
        print("Không tìm thấy df_processed. Vui lòng đảm bảo dữ liệu đã được load và tiền xử lý.")
        print("Để chạy script này độc lập, bỏ comment phần load/tiền xử lý dữ liệu.")
