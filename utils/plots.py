import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def save_plots(y_test, model, predictions, result_path):
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"))
    plt.close()

    # Feature Importance (for tree-based models like XGBoost)
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "feature": [f"feature_{i}" for i in range(len(model.feature_importances_))],
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=feature_importance)
        plt.title("Feature Importance")
        plt.savefig(os.path.join(result_path, "feature_importance.png"))
        plt.close()

def plot_roc_curve(y_test, y_proba, result_path):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_path, "roc_curve.png"))
    plt.close()

def plot_precision_recall_curve(y_test, y_proba, result_path):
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="green", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(result_path, "precision_recall_curve.png"))
    plt.close()


def plot_training_time(runtimes, result_path):
    plt.figure(figsize=(8, 6))
    plt.bar(runtimes.keys(), runtimes.values(), color=['blue', 'orange'])
    plt.xlabel("Model")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time Comparison")
    plt.savefig(os.path.join(result_path, "training_time_comparison.png"))
    plt.close()

