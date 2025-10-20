# ===== Build stage: train and bake model =====
FROM python:3.11-slim AS build

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ model/
# Train v0.1 by default (swap to train_v02.py for v0.2 release)
RUN python model/train_v01.py

# ===== Run stage: serve API with baked model =====
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install only runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model artifacts from builder
COPY --from=build /app/model/model_v01.joblib /app/model/model_v01.joblib
COPY app/ app/

ENV MODEL_PATH=/app/model/model_v01.joblib
ENV MODEL_VERSION=v0.1
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD \
  curl -s http://localhost:8080/health | grep '"status":"ok"' || exit 1

CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8080"]
