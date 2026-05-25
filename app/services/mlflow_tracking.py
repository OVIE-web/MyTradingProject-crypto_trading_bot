"""Optional MLflow tracking helpers for model training runs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.env_loader import load_environment

LOG = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "oviex-quant-engine"


@dataclass(frozen=True, slots=True)
class MLflowSettings:
    """Runtime settings for optional MLflow experiment tracking."""

    enabled: bool = False
    tracking_uri: str = DEFAULT_TRACKING_URI
    experiment_name: str = DEFAULT_EXPERIMENT_NAME

    @classmethod
    def from_env(cls) -> MLflowSettings:
        """Build MLflow settings from environment variables."""
        load_environment()
        enabled = os.getenv("MLFLOW_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            enabled=enabled,
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
        )


def is_mlflow_enabled(settings: MLflowSettings | None = None) -> bool:
    """Return whether MLflow tracking should run."""
    return (settings or MLflowSettings.from_env()).enabled


def log_training_run(
    *,
    model: Any,
    metadata: dict[str, Any],
    model_path: str | Path | None = None,
    settings: MLflowSettings | None = None,
) -> dict[str, Any]:
    """Log a completed model training run to MLflow when enabled.

    This function is intentionally fail-open: MLflow errors are captured in the
    returned metadata instead of breaking model training.
    """
    mlflow_settings = settings or MLflowSettings.from_env()
    if not mlflow_settings.enabled:
        return {}

    try:
        import mlflow
    except ImportError:
        LOG.warning("MLflow tracking requested, but mlflow is not installed.")
        return {"mlflow_error": "mlflow_not_installed"}

    try:
        mlflow.set_tracking_uri(mlflow_settings.tracking_uri)
        mlflow.set_experiment(mlflow_settings.experiment_name)

        run_name = str(metadata.get("model_name", "training_run"))
        with mlflow.start_run(run_name=run_name) as run:
            params = metadata.get("params")
            if isinstance(params, dict):
                mlflow.log_params(_stringify_params(params))

            accuracy = metadata.get("accuracy")
            if isinstance(accuracy, int | float):
                mlflow.log_metric("accuracy", float(accuracy))

            evaluation = metadata.get("evaluation")
            if isinstance(evaluation, dict):
                _log_evaluation_metrics(mlflow, evaluation)
                mlflow.log_dict(_json_safe(evaluation), "evaluation.json")

            mlflow.log_dict(_json_safe(metadata), "model_metadata.json")

            if model_path is not None and Path(model_path).exists():
                mlflow.log_artifact(str(model_path), artifact_path="model")

            _try_log_xgboost_model(mlflow, model)

            return {
                "mlflow_run_id": run.info.run_id,
                "mlflow_artifact_uri": run.info.artifact_uri,
                "mlflow_experiment_name": mlflow_settings.experiment_name,
                "mlflow_tracking_uri": mlflow_settings.tracking_uri,
            }
    except Exception as exc:
        LOG.exception("MLflow training tracking failed: %s", exc)
        return {"mlflow_error": str(exc)}


def _log_evaluation_metrics(mlflow_module: Any, evaluation: dict[str, Any]) -> None:
    accuracy = evaluation.get("accuracy")
    if isinstance(accuracy, int | float):
        mlflow_module.log_metric("evaluation_accuracy", float(accuracy))

    report = evaluation.get("classification_report")
    if not isinstance(report, dict):
        return

    for label, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name in ("precision", "recall", "f1-score"):
            value = metrics.get(metric_name)
            if isinstance(value, int | float):
                safe_label = str(label).replace(" ", "_")
                safe_metric = metric_name.replace("-", "_")
                mlflow_module.log_metric(f"{safe_label}_{safe_metric}", float(value))


def _try_log_xgboost_model(mlflow_module: Any, model: Any) -> None:
    try:
        mlflow_module.xgboost.log_model(model, artifact_path="xgboost_model")
    except Exception as exc:
        LOG.debug("Skipping MLflow XGBoost model logging: %s", exc)


def _stringify_params(params: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in params.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_TRACKING_URI",
    "MLflowSettings",
    "is_mlflow_enabled",
    "log_training_run",
]
