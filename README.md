virtual-diabetes-triage/
├─ app/
│  ├─ main.py                 # FastAPI service: /health and /predict
│  ├─ schema.py               # Pydantic models for request/response
├─ model/
│  ├─ train_v01.py            # v0.1: StandardScaler + LinearRegression
│  ├─ train_v02.py            # v0.2: Ridge or RandomForestRegressor
│  ├─ metrics_v01.json
│  ├─ metrics_v02.json
│  ├─ model_v01.joblib
│  ├─ model_v02.joblib
├─ tests/
│  ├─ test_api.py             # smoke tests for the API
│  ├─ test_training.py        # unit tests for training pipeline
├─ .github/workflows/
│  ├─ ci.yml                  # PR/push checks (lint, tests, quick train, artifacts)
│  ├─ release.yml             # tag build (v*): build, test, push GHCR, GitHub Release
├─ Dockerfile
├─ requirements.txt
├─ CHANGELOG.md
├─ README.md
