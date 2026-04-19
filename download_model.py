import sys
import os
import mlflow

THRESHOLD = 0.85
## you may need to rerun training Pipeline
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(tracking_uri)
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Assignment3_MarkSamuel")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.D_accuracy DESC"],
    max_results=1,
)
if not runs:
    print("\nFAILED — No runs found in the experiment.")
    sys.exit(1)

best_run_id = runs[0].info.run_id
run_info = client.get_run(best_run_id).info
print(f"Actual Artifact Location: {run_info.artifact_uri}, {tracking_uri}")

accuracy = runs[0].data.metrics.get("D_accuracy")

if accuracy < THRESHOLD:
    print(f"\nFAILED — Accuracy {accuracy:.4f} is below the threshold of {THRESHOLD}. ")
    sys.exit(1)
else:
    print(f"\nPASSED — Accuracy {accuracy:.4f} meets the threshold of {THRESHOLD}. ")
    # download the model
    model_uri = f"runs:/{best_run_id}/data" 
    mlflow.pytorch.load_model(model_uri)
    sys.exit(0)
