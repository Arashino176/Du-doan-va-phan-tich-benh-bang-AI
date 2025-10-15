import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

def load_data(disease_type='heart'):
    """
    Load dữ liệu theo loại bệnh được chọn
    """
    if disease_type == 'heart':
        # Load Heart Disease dataset
        data = pd.read_csv('Data/heart.csv')
        X = data.drop('target', axis=1)
        y = data['target']
        feature_names = X.columns.tolist()
        
    elif disease_type == 'diabetes':
        # Load Diabetes dataset
        data = pd.read_csv('Data/diabetes.csv')
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        feature_names = X.columns.tolist()
        
    elif disease_type == 'breast_cancer':
        # Load Breast Cancer dataset từ sklearn
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        feature_names = data.feature_names.tolist()
        
    else:
        raise ValueError("disease_type phải là 'heart', 'diabetes' hoặc 'breast_cancer'")
        
    return X, y, feature_names

def handle_missing_values(X):
    """
    Xử lý missing values trong dataset
    """
    # Tạo bản sao để tránh warning về inplace operation
    X_clean = X.copy()
    # Thay thế missing values bằng median của cột
    for column in X_clean.columns:
        X_clean[column] = X_clean[column].fillna(X_clean[column].median())
    return X_clean

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Tiền xử lý dữ liệu và chia train/test
    """
    # Xử lý missing values
    X = handle_missing_values(X)
    
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Chuẩn hóa features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Chuyển về DataFrame để giữ tên các cột
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler