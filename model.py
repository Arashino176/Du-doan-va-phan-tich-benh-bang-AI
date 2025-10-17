import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import shap

class DiseasePredictor:
    def __init__(self, n_estimators=100, random_state=42, learning_rate=0.1, max_depth=3):
        """
        Khởi tạo mô hình với GradientBoostingClassifier
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Số lượng cây (boosting stages) để thực hiện.
        random_state : int, default=42
            Seed để đảm bảo kết quả có thể tái tạo lại được
        learning_rate : float, default=0.1
            Tốc độ học của mô hình.
        max_depth : int, default=3
            Độ sâu tối đa của mỗi cây.
        """
        # Thay đổi từ RandomForestClassifier sang GradientBoostingClassifier
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            learning_rate=learning_rate,
            max_depth=max_depth
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.explainer = None

    def preprocess_data(self, X):
        """
        Chuẩn hóa dữ liệu đầu vào
        
        Parameters:
        -----------
        X : array-like
            Dữ liệu đầu vào cần chuẩn hóa
            
        Returns:
        --------
        array-like
            Dữ liệu đã được chuẩn hóa
        """
        return self.scaler.transform(X)

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Huấn luyện mô hình
        
        Parameters:
        -----------
        X : array-like
            Features đầu vào
        y : array-like
            Target labels
        test_size : float, default=0.2
            Tỉ lệ chia tập test
        random_state : int, default=42
            Seed cho việc chia tập train/test
            
        Returns:
        --------
        dict
            Dictionary chứa metrics đánh giá mô hình
        """
        # Lưu tên các features và chuyển X về dạng numpy array
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        elif hasattr(X, 'feature_names_in_'):
            self.feature_names = X.feature_names_in_.tolist()
        
        # Chia tập train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Chuẩn hóa dữ liệu
        self.scaler.fit(X_train)
        X_train_scaled = self.preprocess_data(X_train)
        X_test_scaled = self.preprocess_data(X_test)

        # Huấn luyện mô hình
        self.model.fit(X_train_scaled, y_train)

        # Dự đoán
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Tính các metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        # Khởi tạo SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        return metrics

    def predict(self, X):
        """
        Dự đoán cho dữ liệu mới
        
        Parameters:
        -----------
        X : array-like
            Dữ liệu cần dự đoán
            
        Returns:
        --------
        array
            Nhãn dự đoán
        array
            Xác suất dự đoán
        """
        X_scaled = self.preprocess_data(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return predictions, probabilities

    def explain_prediction(self, X):
        """
        Giải thích dự đoán sử dụng SHAP
        
        Parameters:
        -----------
        X : array-like
            Dữ liệu cần giải thích
            
        Returns:
        --------
        array
            SHAP values cho dự đoán
        """
        if self.explainer is None:
            raise ValueError("Mô hình chưa được huấn luyện. Hãy gọi phương thức train() trước.")
        
        X_scaled = self.preprocess_data(X)
        shap_values = self.explainer.shap_values(X_scaled)
        
        return shap_values

    def get_feature_importance(self):
        """
        Lấy độ quan trọng của các features
        
        Returns:
        --------
        dict
            Dictionary chứa tên feature và độ quan trọng tương ứng
        """
        if self.feature_names is None:
            raise ValueError("Không tìm thấy tên các features. Hãy đảm bảo train mô hình với DataFrame có tên cột.")
        
        importances = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importances))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))