# mlflow/Dockerfile
# Minimal MLflow tracking server image (SQLite backend store, local
# filesystem artifact store, both on a mounted volume). Not intended for
# multi-writer production use as-is — a real deployment would point
# --backend-store-uri at a managed Postgres and --default-artifact-root
# at S3/GCS. This is deliberately the simplest correct thing that
# demonstrates the pattern: a dedicated tracking server training runs log
# to, separate from any single training script's local ./mlruns.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install "mlflow>=2.14"

RUN mkdir -p /mlflow/artifacts
WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts"]
