FROM python:3.10-slim


# Add these near the top of your Dockerfile
ARG MLFLOW_TRACKING_URI
ARG MLFLOW_TRACKING_USERNAME
ARG MLFLOW_TRACKING_PASSWORD

# Set them as ENV so the "uv run" command can see them
ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
ENV MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME
ENV MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml . 
COPY uv.lock .

RUN uv sync --frozen --no-install-project --no-dev

COPY . .

RUN ["uv", "run", "download_model.py"]

CMD ["uv", "run", "python", "-c", "print('Model container running.')" ]