import json, os
from model.train_v01 import train_and_eval

def test_train_v01_writes_metrics(tmp_path, monkeypatch):
    out = tmp_path / "model"
    monkeypatch.chdir(tmp_path)
    metrics = train_and_eval(output_dir=str(out))
    assert "rmse" in metrics
    assert (out / "metrics_v01.json").exists()
