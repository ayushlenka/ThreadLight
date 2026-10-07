# Production image: runtime code and dependencies only.
# Tests, evals, and dev tools stay in the repo but never enter the image (.dockerignore,
# and the [dev] extra isn't installed).

FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install .

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 threadlight
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY alembic.ini ./
COPY alembic ./alembic
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER threadlight
ENTRYPOINT ["threadlight"]
CMD ["check"]
