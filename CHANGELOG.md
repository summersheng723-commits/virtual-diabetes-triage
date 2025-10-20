# Changelog

## v0.2
- Model: Ridge (StandardScaler + Ridge(alpha=1.0)) replacing LinearRegression.
- Why: reduces overfitting and improves RMSE on held-out split.
- Metrics (example — update with your actual numbers):
  - v0.1 RMSE: 55.1
  - v0.2 RMSE: 53.6  (Δ -1.5)
- API unchanged; image size unchanged; same port; same schema.

## v0.1
- Baseline pipeline: StandardScaler + LinearRegression.
- Exposed /health and /predict.
- Docker image with baked model (MODEL_VERSION=v0.1).
- CI: lint, tests, quick training smoke; release: build, smoke test, push to GHCR, GitHub Release.
