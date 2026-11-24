from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def save_training_curves(history, output_dir, run_name):
    output_dir = Path(output_dir)
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist.get("loss", []), label="Training loss")
    axes[0].plot(hist.get("val_loss", []), label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(hist.get("accuracy", []), label="Training accuracy")
    axes[1].plot(hist.get("val_accuracy", []), label="Validation accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{run_name}_training_curves.png", dpi=160)
    plt.close(fig)

def evaluate_model(model, test_ds, y_true, class_names, output_dir, run_name):
    output_dir = Path(output_dir)
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    labels = list(range(len(class_names)))

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        zero_division=0, digits=4
    )
    (output_dir / f"{run_name}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(max(9, len(class_names) * 0.45), max(7, len(class_names) * 0.4)))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {run_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / f"{run_name}_confusion_matrix.png", dpi=180)
    plt.close()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "run": run_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }
