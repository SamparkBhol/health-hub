FROM node:22-alpine@sha256:16e22a550f3863206a3f701448c45f7912c6896a62de43add43bb9c86130c3e2 AS web-build

WORKDIR /web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci
COPY apps/web/ ./
RUN mkdir -p public/data
COPY data/boundaries/odisha_districts_census_2011.geojson \
    public/data/odisha_districts_census_2011.geojson
ENV VITE_API_BASE_URL=""
RUN npm run build


FROM ghcr.io/astral-sh/uv:0.11.6@sha256:b1e699368d24c57cda93c338a57a8c5a119009ba809305cc8e86986d4a006754 AS uv-binaries
FROM caddy:2-alpine@sha256:5f5c8640aae01df9654968d946d8f1a56c497f1dd5c5cda4cf95ab7c14d58648 AS caddy-binary


FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS python-deps

ENV UV_LINK_MODE=copy
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv-binaries /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project --frozen --extra nlp


FROM python-deps AS model-build

ENV PATH="/app/.venv/bin:$PATH" \
    ODISHA_MODELS_DIR=/app/models
WORKDIR /app
COPY packages/ /app/packages/
COPY scripts/fetch_models.py /app/scripts/fetch_models.py
RUN python scripts/fetch_models.py


FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    ODISHA_MODELS_DIR=/app/models \
    ODISHA_NLP_THREADS=2 \
    ODISHA_LLM_THREADS=2 \
    DATABASE_URL=sqlite:////app/runtime/odisha_health_hub.db \
    DEPLOYMENT_PROFILE=public_competition \
    AUTO_REPLAY_FIXTURES=false \
    PUBLIC_WRITE_ENABLED=false \
    LIVE_COLLECTION_ENABLED=true \
    ENABLE_IN_PROCESS_SCHEDULER=true \
    COLLECTOR_JOBS_PER_TICK=40 \
    COLLECTOR_FETCH_WORKERS=4 \
    ALLOWED_ORIGINS="*" \
    CRAWLER_CONTACT="+https://github.com/SamparkBhol/health-hub; public-health-evidence-index"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl poppler-utils tesseract-ocr tesseract-ocr-eng \
       tesseract-ocr-hin tesseract-ocr-ori \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-binaries /uv /uvx /bin/
COPY --from=caddy-binary /usr/bin/caddy /usr/bin/caddy
WORKDIR /app
COPY --from=python-deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock ./
COPY --from=model-build /app/models /app/models
COPY . .
COPY --from=web-build /web/dist /app/apps/web/dist

# Models are public and ungated. The separate model-build layer stays cached
# across ordinary application changes and avoids a multi-gigabyte download
# every time free hardware wakes from sleep.

RUN chmod 0755 /app/scripts/start_huggingface.sh \
    && python scripts/doctor.py --runtime \
    && python scripts/make_synthetic.py \
    && useradd --create-home --uid 1000 user \
    && mkdir -p /app/runtime \
    && chown -R user:user /app

USER user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD curl --fail http://127.0.0.1:7860/api/v1/readyz || exit 1

CMD ["/app/scripts/start_huggingface.sh"]
