import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap
import lime
from lime import lime_tabular

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Vẽ confusion matrix với seaborn
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks + 0.5, class_names)
        plt.yticks(tick_marks + 0.5, class_names)
    
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.tight_layout()
    
def plot_roc_curve(y_true, y_pred_proba):
    """
    Vẽ đường cong ROC
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)

def plot_feature_importance(model, feature_names):
    """
    Vẽ biểu đồ độ quan trọng của features
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Độ quan trọng của features')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), 
               [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.tight_layout()

def plot_shap_summary(model, X, feature_names=None):
    """
    Vẽ SHAP summary plot để giải thích model
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()

def explain_prediction_lime(model, X_train, X_test_instance, feature_names, class_names):
    """
    Sử dụng LIME để giải thích một dự đoán cụ thể
    """
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    explanation = explainer.explain_instance(
        X_test_instance, 
        model.predict_proba,
        num_features=len(feature_names)
    )
    
    return explanation

def display_metrics(metrics_dict):
    """
    Hiển thị các metrics đánh giá
    """
    print("\n=== Model Performance Metrics ===")
    print(f"Accuracy: {metrics_dict['accuracy']:.3f}")
    print(f"Precision: {metrics_dict['precision']:.3f}")
    print(f"Recall: {metrics_dict['recall']:.3f}")
    print(f"F1-score: {metrics_dict['f1']:.3f}")
    print(f"ROC AUC: {metrics_dict['roc_auc']:.3f}")
    print("================================")