FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml . 
COPY uv.lock .

RUN uv sync --frozen --no-install-project --no-dev

COPY . .

RUN ["uv", "run", "download_model.py"]

CMD ["uv", "run", "python", "-c", "print('Model container running.')" ]