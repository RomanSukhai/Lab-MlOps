import argparse
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from data import load_data
from preprocessing import preprocess_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest model")

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default="Telco_Churn_Experiment")
    parser.add_argument("--author", type=str, default="Roman")

    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = importances.argsort()[-15:]  # топ 15

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()


def train():

    args = parse_args()

    df = load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():

        # ===== TAGS =====
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset_version", "v1")

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)

        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)

        mlflow.sklearn.log_model(model, "model")

        plot_confusion_matrix(y_test, y_test_pred)
        plot_feature_importance(model, X_train.columns)

        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("feature_importance.png")

        print("Training finished.")
        print(f"Train Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    train()
