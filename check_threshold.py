import sys
import os
import mlflow

THRESHOLD = 0.85


with open("model_info.txt", "r") as f:
    run_id = f.read().strip()


print(f"Checking accuracy for Run ID: {run_id}")

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(tracking_uri)
client = mlflow.tracking.MlflowClient()


run = client.get_run(run_id)

accuracy = run.data.metrics.get("D_accuracy")

if accuracy < THRESHOLD:
    print(f"\nFAILED — Accuracy {accuracy:.4f} is below the threshold of {THRESHOLD}. ")
    sys.exit(1)
else:
    print(f"\nPASSED — Accuracy {accuracy:.4f} meets the threshold of {THRESHOLD}. ")
    sys.exit(0)
