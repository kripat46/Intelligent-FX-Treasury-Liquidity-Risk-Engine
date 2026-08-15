# dashboard/Dockerfile
# Build from the REPO ROOT, since it needs both dashboard/ and src/:
#
#   docker build -f dashboard/Dockerfile -t treasury-dashboard:latest .
#
# NOTE: not built/pushed from this environment — see README's "What's
# verified vs. not" section.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt requirements-dashboard.txt ./
RUN pip install -r requirements-dashboard.txt

COPY src/ ./src/
COPY dashboard/ ./dashboard/

RUN mkdir -p /app/data
ENV DATA_DIR=/app/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
