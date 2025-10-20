import json, os, joblib, numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from datetime import datetime
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def load_data():
    Xy = load_diabetes(as_frame=True)
    df = Xy.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def train_and_eval(output_dir="model"):
    X, y = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    rmse = float(mean_squared_error(yte, preds, squared=False))

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model_v01.joblib")
    joblib.dump({"pipeline": pipe, "trained_at": datetime.utcnow().isoformat()}, model_path)

    metrics = {"rmse": rmse, "model_version": "v0.1", "seed": SEED}
    with open(os.path.join(output_dir, "metrics_v01.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics))
    return metrics

if __name__ == "__main__":
    train_and_eval()
