FROM python:3.14-slim

# EXTRAS selects what to install: "tracking" is enough for the MLflow server.
ARG EXTRAS=tracking
# Set to 1 if a dependency has no wheel for your platform and must be compiled.
ARG WITH_COMPILER=0

# libgomp1: OpenMP runtime needed by implicit / CatBoost wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    $( [ "$WITH_COMPILER" = "1" ] && echo build-essential ) \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# requirements.loc[k]: the glob copies the lock file only if it exists
COPY pyproject.toml README.md requirements.loc[k] ./
COPY src ./src
COPY configs ./configs

RUN if [ -f requirements.lock ]; then \
        pip install --no-cache-dir -c requirements.lock ".[${EXTRAS}]"; \
    else \
        pip install --no-cache-dir ".[${EXTRAS}]"; \
    fi

ENV PYTHONUNBUFFERED=1 RECOPS_CONFIG_DIR=/app/configs
