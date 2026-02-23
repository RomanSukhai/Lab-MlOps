import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split


def preprocess_data(df: pd.DataFrame):
    # Видаляємо customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges іноді з пробілами
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Заповнюємо пропуски
    df = df.fillna(df.median(numeric_only=True))

    # Target encoding
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    # Аргументи з командного рядка
    input_path = sys.argv[1]      # data/raw/dataset.csv
    output_dir = sys.argv[2]      # data/prepared

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Зберігаємо разом X і y
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("Data preprocessing completed successfully.")