import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from transform import clean_data


def train_model(data_path: str):
    df = pd.read_csv(data_path)
    df = clean_data(df)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print("Accuracy", acc)
