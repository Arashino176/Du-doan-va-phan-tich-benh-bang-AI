from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import joblib
from utils import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_shap_summary,
    explain_prediction_lime,
    display_metrics
)

def evaluate_model(model, X_test, y_test, feature_names):
    """
    Đánh giá mô hình và tạo báo cáo
    """
    # Dự đoán
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Tính toán các metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Hiển thị metrics
    display_metrics(metrics)
    
    # Vẽ confusion matrix
    plot_confusion_matrix(y_test, y_pred, ['Negative', 'Positive'])
    
    # Vẽ ROC curve
    plot_roc_curve(y_test, y_pred_proba)
    
    # Vẽ feature importance
    plot_feature_importance(model, feature_names)
    
    return metrics

def analyze_with_shap(model, X, feature_names):
    """
    Phân tích model bằng SHAP
    """
    # Vẽ SHAP summary plot
    plot_shap_summary(model, X, feature_names)

def explain_predictions_with_lime(model, X_train, X_test, feature_names, n_samples=5):
    """
    Giải thích các dự đoán bằng LIME
    """
    # Chọn ngẫu nhiên một số mẫu để giải thích
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    for idx in indices:
        print(f"\nGiải thích mẫu thứ {idx}:")
        explanation = explain_prediction_lime(
            model,
            X_train.values if hasattr(X_train, 'values') else X_train,
            X_test.iloc[idx].values if hasattr(X_test, 'iloc') else X_test[idx],
            feature_names,
            ['Negative', 'Positive']
        )
        # Hiển thị giải thích dạng text
        print("\nFeature importance:")
        for feat, imp in explanation.as_list():
            print(f"{feat}: {imp:.4f}")
        print("\n" + "="*50)

def save_evaluation_report(metrics, filename):
    """
    Lưu báo cáo đánh giá
    """
    with open(filename, 'w') as f:
        f.write("=== Model Evaluation Report ===\n\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.3f}\n")
    print(f"Đã lưu báo cáo đánh giá vào {filename}")

def save_model(model, filename):
    """
    Lưu mô hình đã train
    """
    joblib.dump(model, filename)
    print(f"Đã lưu mô hình vào {filename}")

def load_model(filename):
    """
    Load mô hình đã lưu
    """
    return joblib.load(filename)