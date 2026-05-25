"""Unit tests for the MLflow tracking integration in app.services.mlflow_tracking."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.services.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    MLflowSettings,
    is_mlflow_enabled,
    log_training_run,
)


class FakeRun:
    def __init__(self) -> None:
        self.info = SimpleNamespace(run_id="run-123", artifact_uri="file:///tmp/mlruns/run-123")

    def __enter__(self) -> FakeRun:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class FakeXGBoostModule:
    def __init__(self) -> None:
        self.logged_model: object | None = None

    def log_model(self, model: object, *, artifact_path: str) -> None:
        self.logged_model = (model, artifact_path)


class FakeMLflowModule:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.params: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.dict_artifacts: dict[str, Any] = {}
        self.artifacts: list[tuple[str, str | None]] = []
        self.xgboost = FakeXGBoostModule()

    def set_tracking_uri(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def start_run(self, *, run_name: str) -> FakeRun:
        assert run_name == "xgboost_signal_model"
        return FakeRun()

    def log_params(self, params: dict[str, str]) -> None:
        self.params.update(params)

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = value

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        self.dict_artifacts[artifact_file] = dictionary

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((local_path, artifact_path))


def test_mlflow_settings_from_env_reads_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)

    settings = MLflowSettings.from_env()

    assert settings.enabled is False
    assert settings.tracking_uri == DEFAULT_TRACKING_URI
    assert settings.experiment_name == DEFAULT_EXPERIMENT_NAME


def test_mlflow_settings_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "custom-experiment")

    settings = MLflowSettings.from_env()

    assert settings.enabled is True
    assert settings.tracking_uri == "sqlite:///mlflow.db"
    assert settings.experiment_name == "custom-experiment"
    assert is_mlflow_enabled(settings) is True


def test_log_training_run_returns_empty_when_disabled() -> None:
    result = log_training_run(
        model=object(),
        metadata={"model_name": "xgboost_signal_model"},
        settings=MLflowSettings(enabled=False),
    )

    assert result == {}


def test_log_training_run_records_metadata_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_mlflow = FakeMLflowModule()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    model_path = tmp_path / "xgboost_model.json"
    model_path.write_text("{}", encoding="utf-8")

    result = log_training_run(
        model=object(),
        model_path=model_path,
        metadata={
            "model_name": "xgboost_signal_model",
            "accuracy": 0.91,
            "params": {"max_depth": 3, "eval_metric": "mlogloss"},
            "evaluation": {
                "accuracy": 0.91,
                "classification_report": {
                    "buy": {
                        "precision": 0.9,
                        "recall": 0.8,
                        "f1-score": 0.85,
                    }
                },
            },
        },
        settings=MLflowSettings(
            enabled=True,
            tracking_uri="sqlite:///mlflow.db",
            experiment_name="test-experiment",
        ),
    )

    assert result["mlflow_run_id"] == "run-123"
    assert result["mlflow_artifact_uri"] == "file:///tmp/mlruns/run-123"
    assert fake_mlflow.tracking_uri == "sqlite:///mlflow.db"
    assert fake_mlflow.experiment_name == "test-experiment"
    assert fake_mlflow.params["max_depth"] == "3"
    assert fake_mlflow.metrics["accuracy"] == 0.91
    assert fake_mlflow.metrics["buy_f1_score"] == 0.85
    assert "evaluation.json" in fake_mlflow.dict_artifacts
    assert (str(model_path), "model") in fake_mlflow.artifacts
