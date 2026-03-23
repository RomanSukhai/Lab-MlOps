import mlflow
import optuna
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


SEED = 42


def load_data():

    train_df = pd.read_csv("data/prepared/train.csv")
    test_df = pd.read_csv("data/prepared/test.csv")

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]

    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    return X_train, X_test, y_train, y_test


def objective(trial):

    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 15)

    with mlflow.start_run(nested=True):

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=SEED
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("f1", f1)

        return f1


if __name__ == "__main__":

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("HPO_Telco_Churn")

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name="optuna_parent"):

        sampler = optuna.samplers.TPESampler(seed=SEED)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )

        study.optimize(objective, n_trials=20)

        best_params = study.best_trial.params
        best_score = study.best_trial.value

        mlflow.log_metric("best_f1", best_score)
        mlflow.log_params(best_params)

        best_model = RandomForestClassifier(
            **best_params,
            random_state=SEED
        )

        best_model.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)

        joblib.dump(best_model, "models/best_model.pkl")

        mlflow.sklearn.log_model(best_model, "best_model")

        print("Best params:", best_params)
        print("Best F1:", best_score)
        