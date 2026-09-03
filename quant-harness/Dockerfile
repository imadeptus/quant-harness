# quant-harness verdict API — multi-stage, non-root, SIGTERM-friendly.
#
#   docker build -t qh-api:dev .
#   docker run --rm -p 8000:8000 -e QH_API_KEYS=changeme qh-api:dev
#
# The `api` extra (fastapi, uvicorn, pydantic) is declared in pyproject.toml.

ARG PYTHON_VERSION=3.12

# ---- build stage: install the package + API deps into an isolated prefix --------
FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY harness ./harness
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install ".[api]"

# ---- runtime stage --------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000
RUN groupadd --system app && useradd --system --gid app --uid 10001 --create-home app
COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=app:app api ./api
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz', timeout=2).status==200 else 1)"
# uvicorn handles SIGTERM itself (graceful shutdown); `sh -c` only expands $PORT.
# --no-access-log: the app already writes one JSON line per request to stdout.
CMD ["sh", "-c", "exec python -m uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log"]
