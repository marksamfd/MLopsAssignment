# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Build argument: MLflow Run ID passed in at build time ─────────────────────
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Simulate model download from MLflow ──────────────────────────────────────
# In production, replace the echo with an mlflow artifacts download command:
#   RUN mlflow artifacts download -r ${RUN_ID} -d /app/model
RUN echo "Downloading model for Run ID: ${RUN_ID}" && \
    mkdir -p /app/model && \
    echo "${RUN_ID}" > /app/model/run_id.txt && \
    echo "Model download complete (simulated)."

# ── Default entrypoint ────────────────────────────────────────────────────────
CMD ["python", "-c", "print('Model f container running. Run ID:', open('/app/model/run_id.txt').read().strip())"]