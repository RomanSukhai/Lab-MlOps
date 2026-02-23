import pandas as pd
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
