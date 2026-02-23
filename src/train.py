import argparse
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest model")

    # DVC arguments
    parser.add_argument("input_dir", type=str)     # data/prepared
    parser.add_argument("output_dir", type=str)    # data/models

    # Model params
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default="Telco_Churn_Experiment")
    parser.add_argument("--author", type=str, default="Roman")

    return parser.parse_args()


def evaluate_model(model, X_test, y_test, output_dir):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    return accuracy, f1, cm_path


def plot_feature_importance(model, feature_names, output_dir):
    importances = model.feature_importances_
    indices = importances.argsort()[-15:]

    fi_path = os.path.join(output_dir, "feature_importance.png")

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(fi_path)
    plt.close()

    return fi_path


def train():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 🔹 Читаємо підготовлені дані
    train_df = pd.read_csv(os.path.join(args.input_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.input_dir, "test.csv"))

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]

    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():

        mlflow.set_tag("author", args.author)
        mlflow.set_tag("model_type", "RandomForest")

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        # ===== TRAIN METRICS =====
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)

        # ===== TEST METRICS + CONF MATRIX =====
        test_acc, test_f1, cm_path = evaluate_model(
            model,
            X_test,
            y_test,
            args.output_dir
        )

        # ===== FEATURE IMPORTANCE =====
        fi_path = plot_feature_importance(
            model,
            X_train.columns,
            args.output_dir
        )

        # ===== LOGGING =====
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)

        mlflow.sklearn.log_model(model, "model")

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)

        print("Training finished.")
        print(f"Train Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    train()