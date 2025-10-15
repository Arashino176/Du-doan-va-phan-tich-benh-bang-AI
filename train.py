import argparse
import os
from preprocess import load_data, preprocess_data
from model import DiseasePredictor
from evaluate import evaluate_model, analyze_with_shap, explain_predictions_with_lime, save_model, save_evaluation_report

def train_pipeline(disease_type, n_estimators=100):
    """
    Pipeline huấn luyện hoàn chỉnh
    """
    print(f"\nBắt đầu huấn luyện mô hình cho bệnh: {disease_type}")
    
    # Define paths
    model_dir = 'model'
    report_dir = 'reports'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    save_path = os.path.join(model_dir, f"{disease_type}_model.joblib")
    report_path = os.path.join(report_dir, f"{disease_type}_evaluation_report.txt")

    # 1. Load dữ liệu
    print("\n1. Loading data...")
    X, y, feature_names = load_data(disease_type)
    print(f"Loaded {X.shape[0]} mẫu với {X.shape[1]} features")
    
    # 2. Tiền xử lý dữ liệu
    print("\n2. Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print("Data đã được xử lý và chia train/test")
    
    # 3. Khởi tạo và huấn luyện mô hình
    print("\n3. Training model...")
    model = DiseasePredictor(n_estimators=n_estimators)
    model.train(X_train, y_train)
    print("Model đã được huấn luyện xong")
    
    # 4. Đánh giá mô hình
    print("\n4. Evaluating model...")
    metrics = evaluate_model(model.model, X_test, y_test, feature_names)
    
    # 5. Phân tích SHAP
    print("\n5. Analyzing with SHAP...")
    analyze_with_shap(model.model, X_test, feature_names)
    
    # 6. Giải thích với LIME
    print("\n6. Explaining predictions with LIME...")
    explain_predictions_with_lime(model.model, X_train, X_test, feature_names)
    
    # 7. Lưu mô hình và báo cáo
    print("\n7. Saving model and evaluation report...")
    save_model(model, save_path)
    save_evaluation_report(metrics, report_path)
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description='Train disease prediction model')
    parser.add_argument('--disease', type=str, default=['heart'], nargs='+',
                      choices=['heart', 'diabetes', 'breast_cancer', 'all'],
                      help='Loại bệnh cần dự đoán. Có thể chọn nhiều bệnh hoặc "all" để huấn luyện tất cả.')
    parser.add_argument('--n_estimators', type=int, default=100,
                      help='Số lượng cây trong Random Forest')
    
    args = parser.parse_args()
    
    diseases_to_train = args.disease
    
    if 'all' in diseases_to_train:
        diseases_to_train = ['heart', 'diabetes', 'breast_cancer']

    for disease in diseases_to_train:
        train_pipeline(
            disease_type=disease,
            n_estimators=args.n_estimators
        )

if __name__ == '__main__':
    main()