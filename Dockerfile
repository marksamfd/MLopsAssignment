FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

COPY . .

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "python", "-c", "print('Model container running. Run ID:', open('/app/model/run_id.txt').read().strip())"]