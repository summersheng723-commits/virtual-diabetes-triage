import json, os, joblib, numpy as np, random
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from datetime import datetime

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
        ("ridge", Ridge(alpha=1.0, random_state=SEED))
    ])
    # Alternative:
    # pipe = Pipeline([("rf", RandomForestRegressor(
    #     n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1))])

    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    rmse = float(mean_squared_error(yte, preds, squared=False))

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model_v02.joblib")
    joblib.dump({"pipeline": pipe, "trained_at": datetime.utcnow().isoformat()}, model_path)

    metrics = {"rmse": rmse, "model_version": "v0.2", "seed": SEED}
    with open(os.path.join(output_dir, "metrics_v02.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics))
    return metrics

if __name__ == "__main__":
    train_and_eval()
