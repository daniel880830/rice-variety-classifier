# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- builder
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install --no-deps .

# Slimming: drop bundled test suites and bytecode caches from site-packages.
ARG SLIM=true
RUN if [ "$SLIM" = "true" ]; then \
        find /install -depth -type d \( -name tests -o -name __pycache__ \) -exec rm -rf {} + ; \
        find /install -name "*.pyi" -delete ; \
    fi

# ---------------------------------------------------------------- runtime
FROM python:3.13-slim AS runtime

ARG SLIM=true
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RICE_API_HOST=0.0.0.0 \
    RICE_API_PORT=8000

# pip/setuptools are not needed at runtime.
RUN if [ "$SLIM" = "true" ]; then \
        pip uninstall -y pip setuptools wheel >/dev/null 2>&1 || true ; \
        rm -rf /usr/local/lib/python3.13/ensurepip ; \
    fi \
    && useradd --create-home --uid 1000 app

COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=app:app config ./config
# The dataset is not baked into the image; docker compose mounts ./data/raw read-only.
RUN mkdir -p data/raw models reports && chown -R app:app data models reports

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

CMD ["rice", "serve"]
