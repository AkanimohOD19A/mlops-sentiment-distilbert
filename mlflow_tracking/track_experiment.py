import os
import sys
import json
import mlflow
import argparse
from mlflow.models import infer_signature
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def register_model(
        model_dir,
        experiment_name="emotion-classification",
        run_name=None,
        tracking_uri=None
):
    """
    Register a trained model with MLflow

    Args:
        model_dir: Directory containing the saved model
        experiment_name: Name of the MLflow experiment
        run_name: Name for this specific run
        tracking_uri: MLflow tracking server URI
    """
    # Set up MLflow tracking
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using MLflow tracking URI: {tracking_uri}")
    else:
        print("Using local MLflow tracking")

    # Load model metadata
    try:
        with open(os.path.join(model_dir, 'label_mappings.json'), 'r') as f:
            mappings = json.load(f)
            label2id = mappings['label2id']
            id2label = {int(k): v for k, v in mappings['id2label'].items()}
    except Exception as e:
        print(f"Error loading model metadata: {e}")
        return

    # Load the model and tokenizer
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    # Set up the experiment
    mlflow.set_experiment(experiment_name)

    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log model parameters
        mlflow.log_param("model_type", model.config.model_type)
        mlflow.log_param("num_labels", model.config.num_labels)
        mlflow.log_param("hidden_size", model.config.hidden_size)
        mlflow.log_param("vocab_size", model.config.vocab_size)

        # Create a sample input for the model signature
        sample_text = "This is a sample text for the model signature"
        sample_inputs = tokenizer(
            sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Create a sample output for the model signature
        with torch.no_grad():
            outputs = model(**sample_inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()

        # Create the model signature
        signature = infer_signature(
            sample_inputs.data,
            {"probabilities": probabilities}
        )

        # Log the model
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": tokenizer
            },
            artifact_path="emotion-classifier",
            task="text-classification",
            signature=signature
        )

        # Log additional information
        mlflow.log_dict(label2id, "label2id.json")
        mlflow.log_dict(id2label, "id2label.json")

        # Get the run ID
        run_id = mlflow.active_run().info.run_id
        print(f"Model registered with MLflow. Run ID: {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a model with MLflow")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the saved model"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="emotion-classification",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this run"
    )
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default=None,
        help="MLflow tracking server URI"
    )

    args = parser.parse_args()
    register_model(
        args.model_dir, args.experiment,
        args.run_name, args.tracking_uri
    )