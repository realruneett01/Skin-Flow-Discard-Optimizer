# syntax=docker/dockerfile:1
# CPU image:  docker build -t skdo .
# GPU image:  docker build -t skdo:gpu --build-arg EXTRAS=gpu .      (run with --gpus all)
FROM python:3.12-slim AS base
ARG EXTRAS=""
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 10001 skdo
WORKDIR /app

# Dependencies first so source edits don't invalidate this layer.
COPY pyproject.toml LICENSE ./
RUN mkdir skinflow_discard_optimizer && touch skinflow_discard_optimizer/__init__.py \
    && pip install ".${EXTRAS:+[$EXTRAS]}" && pip uninstall -y skinflow-discard-optimizer

COPY skinflow_discard_optimizer ./skinflow_discard_optimizer
COPY config ./config
COPY contracts ./contracts
# Editable install: the package locates config/, contracts/ and artifacts/ relative to the source tree.
RUN pip install --no-deps -e . && mkdir -p artifacts data site reports && chown -R skdo:skdo /app

USER skdo
# Models, generated data and the per-press site config live in volumes, not in the image.
VOLUME ["/app/artifacts", "/app/data", "/app/site"]

HEALTHCHECK --interval=60s --timeout=30s --start-period=3600s --retries=3 CMD ["skdo", "doctor", "--quick"]

# `skdo up` = configure from SKDO_* env, size a profile to this host, build missing models, smoke test, run.
ENTRYPOINT ["skdo", "up"]
# Default is a short self-test stream from the digital twin. Replace for real use (see docs/DEPLOYMENT.md).
CMD ["--", "--mock", "--mock-cycles", "20"]
