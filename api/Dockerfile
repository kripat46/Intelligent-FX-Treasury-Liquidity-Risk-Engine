# api/Dockerfile
# Lean image for the FastAPI serving layer only — does not include
# Streamlit/Plotly, which the dashboard image carries separately. Build
# from the REPO ROOT (not this directory), since it needs both api/ and
# src/:
#
#   docker build -f api/Dockerfile -t treasury-api:latest .
#
# NOTE: not built/pushed from this environment — see README's "What's
# verified vs. not" section for exactly what was and wasn't executed.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps needed to build xgboost/scipy wheels on slim base images
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt requirements-api.txt ./
RUN pip install -r requirements-api.txt

COPY src/ ./src/
COPY api/ ./api/

# data/ is intentionally NOT copied in — the app bootstraps its own
# synthetic dataset + trained model on first startup (see src/bootstrap.py),
# so the image stays free of any baked-in "data".
RUN mkdir -p /app/data
ENV DATA_DIR=/app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
