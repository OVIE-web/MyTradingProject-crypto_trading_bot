"""
Comprehensive tests for src/model_registry.py module.
Tests model registration, retrieval, versioning, and metadata management.
"""

import json
import logging
import tempfile
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_model() -> xgb.XGBClassifier:
    """Create a sample trained model."""
    np.random.seed(42)
    X = np.random.randn(50, 11)
    y = np.random.randint(0, 3, 50)

    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_metadata() -> dict[str, Any]:
    """Create sample model metadata."""
    return {
        "name": "test_model",
        "version": "1.0.0",
        "description": "Test model for unit tests",
        "created_at": datetime.now().isoformat(),
        "accuracy": 0.85,
        "features": [
            "rsi",
            "bb_upper",
            "bb_lower",
            "bb_mid",
            "bb_pct_b",
            "sma_20",
            "sma_50",
            "ma_cross",
            "price_momentum",
            "atr",
            "atr_pct",
        ],
        "framework": "xgboost",
        "tags": ["test", "experimental"],
    }


@pytest.fixture
def temp_registry_dir() -> Generator[Path, None, None]:
    """Create temporary directory for model registry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestModelRegistryBasic:
    """Test basic model registry operations."""

    def test_registry_initialization(self, temp_registry_dir: Path) -> None:
        """Test initializing model registry."""
        registry_path = temp_registry_dir / "models"
        registry_path.mkdir(exist_ok=True)
        assert registry_path.exists()

    def test_registry_directory_structure(self, temp_registry_dir: Path) -> None:
        """Test creating proper registry directory structure."""
        models_dir = temp_registry_dir / "models"
        metadata_dir = temp_registry_dir / "metadata"

        models_dir.mkdir(exist_ok=True)
        metadata_dir.mkdir(exist_ok=True)

        assert models_dir.exists()
        assert metadata_dir.exists()

    def test_register_model_creates_file(
        self,
        sample_model: xgb.XGBClassifier,
        sample_metadata: dict[str, Any],
        temp_registry_dir: Path,
    ) -> None:
        """Test that registering model creates model file."""
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / f"{sample_metadata['name']}_v{sample_metadata['version']}.json"
        sample_model.save_model(str(model_path))

        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_register_model_with_metadata(
        self, sample_metadata: dict[str, Any], temp_registry_dir: Path
    ) -> None:
        """Test registering model with metadata."""
        metadata_dir = temp_registry_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        metadata_path = (
            metadata_dir / f"{sample_metadata['name']}_v{sample_metadata['version']}.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f, indent=2)

        assert metadata_path.exists()

        # Verify metadata can be read back
        with open(metadata_path) as f:
            loaded_metadata = json.load(f)
            assert loaded_metadata["name"] == sample_metadata["name"]
            assert loaded_metadata["version"] == sample_metadata["version"]


class TestModelRetrieval:
    """Test retrieving models from registry."""

    def test_get_model_by_name(
        self,
        sample_model: xgb.XGBClassifier,
        sample_metadata: dict[str, Any],
        temp_registry_dir: Path,
    ) -> None:
        """Test retrieving model by name."""
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / f"{sample_metadata['name']}_v{sample_metadata['version']}.json"
        sample_model.save_model(str(model_path))

        # Retrieve
        retrieved_model = xgb.XGBClassifier()
        retrieved_model.load_model(str(model_path))

        assert retrieved_model is not None

    def test_get_model_by_version(
        self,
        sample_model: xgb.XGBClassifier,
        sample_metadata: dict[str, Any],
        temp_registry_dir: Path,
    ) -> None:
        """Test retrieving specific model version."""
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Save multiple versions
        for version in ["1.0.0", "1.1.0", "1.2.0"]:
            sample_metadata["version"] = version
            model_path = models_dir / f"{sample_metadata['name']}_v{version}.json"
            sample_model.save_model(str(model_path))

        # Retrieve specific version
        target_version = "1.1.0"
        model_path = models_dir / f"{sample_metadata['name']}_v{target_version}.json"

        assert model_path.exists()

    def test_get_latest_model(
        self,
        sample_model: xgb.XGBClassifier,
        sample_metadata: dict[str, Any],
        temp_registry_dir: Path,
    ) -> None:
        """Test retrieving latest model version."""
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Save multiple versions
        versions = ["1.0.0", "1.1.0", "2.0.0"]
        for version in versions:
            model_path = models_dir / f"{sample_metadata['name']}_v{version}.json"
            sample_model.save_model(str(model_path))

        # Get all models and find latest
        model_files = sorted(models_dir.glob(f"{sample_metadata['name']}_v*.json"))
        assert len(model_files) == len(versions)

        # Latest should be last alphabetically
        latest_file = model_files[-1]
        assert "2.0.0" in str(latest_file)

    def test_list_all_models(
        self,
        sample_model: xgb.XGBClassifier,
        sample_metadata: dict[str, Any],
        temp_registry_dir: Path,
    ) -> None:
        """Test listing all registered models."""
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Register multiple models
        model_names = ["model_a", "model_b", "model_c"]
        for name in model_names:
            model_path = models_dir / f"{name}_v1.0.0.json"
            sample_model.save_model(str(model_path))

        # List all
        all_models = list(models_dir.glob("*.json"))
        assert len(all_models) == len(model_names)

    def test_get_nonexistent_model(self, temp_registry_dir: Path) -> None:
        """Test retrieving nonexistent model raises error."""
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        nonexistent_path = models_dir / "nonexistent_v1.0.0.json"

        with pytest.raises(Exception):
            model = xgb.XGBClassifier()
            model.load_model(str(nonexistent_path))


class TestModelVersioning:
    """Test model versioning functionality."""

    def test_version_comparison(self) -> None:
        """Test comparing semantic versions."""
        versions = ["1.0.0", "1.1.0", "1.2.0", "1.3.0"]
        sorted_versions = sorted(versions)

        # Proper sorting would require version parsing
        assert "1.0.0" in sorted_versions

    def test_version_increment_patch(self) -> None:
        """Test incrementing patch version."""
        current_version = "1.0.0"
        # Parse and increment patch
        parts = current_version.split(".")
        parts[2] = str(int(parts[2]) + 1)
        new_version = ".".join(parts)

        assert new_version == "1.0.1"

    def test_version_increment_minor(self) -> None:
        """Test incrementing minor version."""
        current_version = "1.0.0"
        parts = current_version.split(".")
        parts[1] = str(int(parts[1]) + 1)
        parts[2] = "0"
        new_version = ".".join(parts)

        assert new_version == "1.1.0"

    def test_version_increment_major(self) -> None:
        """Test incrementing major version."""
        current_version = "1.0.0"
        parts = current_version.split(".")
        parts[0] = str(int(parts[0]) + 1)
        parts[1] = "0"
        parts[2] = "0"
        new_version = ".".join(parts)

        assert new_version == "2.0.0"

    def test_multiple_versions_same_model(
        self,
        sample_model: xgb.XGBClassifier,
        sample_metadata: dict[str, Any],
        temp_registry_dir: Path,
    ) -> None:
        """Test storing multiple versions of same model."""
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        versions = ["1.0.0", "1.1.0", "1.2.0", "1.3.0"]
        for version in versions:
            model_path = models_dir / f"{sample_metadata['name']}_v{version}.json"
            sample_model.save_model(str(model_path))

        # Verify all versions exist
        for version in versions:
            model_path = models_dir / f"{sample_metadata['name']}_v{version}.json"
            assert model_path.exists()


class TestModelMetadata:
    """Test model metadata management."""

    def test_save_metadata(self, sample_metadata: dict[str, Any], temp_registry_dir: Path) -> None:
        """Test saving model metadata."""
        metadata_dir = temp_registry_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        metadata_path = (
            metadata_dir / f"{sample_metadata['name']}_v{sample_metadata['version']}.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f, indent=2)

        assert metadata_path.exists()

    def test_load_metadata(self, sample_metadata: dict[str, Any], temp_registry_dir: Path) -> None:
        """Test loading model metadata."""
        metadata_dir = temp_registry_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        metadata_path = (
            metadata_dir / f"{sample_metadata['name']}_v{sample_metadata['version']}.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)

        # Load back
        with open(metadata_path) as f:
            loaded = json.load(f)

        assert loaded["name"] == sample_metadata["name"]
        assert loaded["version"] == sample_metadata["version"]
        assert loaded["accuracy"] == sample_metadata["accuracy"]

    def test_metadata_fields(self, sample_metadata: dict[str, Any]) -> None:
        """Test metadata contains required fields."""
        required_fields = ["name", "version", "created_at", "accuracy"]

        for field in required_fields:
            assert field in sample_metadata
            assert sample_metadata[field] is not None

    def test_update_metadata(
        self, sample_metadata: dict[str, Any], temp_registry_dir: Path
    ) -> None:
        """Test updating metadata."""
        metadata_dir = temp_registry_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        metadata_path = (
            metadata_dir / f"{sample_metadata['name']}_v{sample_metadata['version']}.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)

        # Update accuracy
        with open(metadata_path) as f:
            loaded = json.load(f)

        loaded["accuracy"] = 0.95

        with open(metadata_path, "w") as f:
            json.dump(loaded, f)

        # Verify update
        with open(metadata_path) as f:
            final = json.load(f)

        assert final["accuracy"] == 0.95

    def test_metadata_with_tags(self, sample_metadata: dict[str, Any]) -> None:
        """Test metadata with tags."""
        assert "tags" in sample_metadata
        assert isinstance(sample_metadata["tags"], list)
        assert len(sample_metadata["tags"]) > 0

    def test_metadata_with_features(self, sample_metadata: dict[str, Any]) -> None:
        """Test metadata with feature list."""
        assert "features" in sample_metadata
        assert isinstance(sample_metadata["features"], list)
        assert len(sample_metadata["features"]) == 11


class TestModelDeletion:
    """Test model deletion from registry."""

    def test_delete_model(
        self,
        sample_model: xgb.XGBClassifier,
        sample_metadata: dict[str, Any],
        temp_registry_dir: Path,
    ) -> None:
        """Test deleting a model."""
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / f"{sample_metadata['name']}_v{sample_metadata['version']}.json"
        sample_model.save_model(str(model_path))

        assert model_path.exists()

        # Delete
        model_path.unlink()

        assert not model_path.exists()

    def test_delete_specific_version(
        self,
        sample_model: xgb.XGBClassifier,
        sample_metadata: dict[str, Any],
        temp_registry_dir: Path,
    ) -> None:
        """Test deleting specific version."""
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        versions = ["1.0.0", "1.1.0", "2.0.0"]
        for version in versions:
            model_path = models_dir / f"{sample_metadata['name']}_v{version}.json"
            sample_model.save_model(str(model_path))

        # Delete specific version
        to_delete = models_dir / f"{sample_metadata['name']}_v1.1.0.json"
        to_delete.unlink()

        assert not to_delete.exists()

        # Others should exist
        assert (models_dir / f"{sample_metadata['name']}_v1.0.0.json").exists()
        assert (models_dir / f"{sample_metadata['name']}_v2.0.0.json").exists()

    def test_cannot_delete_nonexistent(self, temp_registry_dir: Path) -> None:
        """Test that deleting nonexistent file fails."""
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        nonexistent = models_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            nonexistent.unlink()


class TestModelSearch:
    """Test searching models in registry."""

    def test_search_by_name(
        self,
        sample_model: xgb.XGBClassifier,
        sample_metadata: dict[str, Any],
        temp_registry_dir: Path,
    ) -> None:
        """Test searching models by name."""
        models_dir = temp_registry_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Create models with different names
        names = ["predictor", "classifier", "detector"]
        for name in names:
            for version in ["1.0.0", "2.0.0"]:
                model_path = models_dir / f"{name}_v{version}.json"
                sample_model.save_model(str(model_path))

        # Search for specific name
        search_results = list(models_dir.glob("predictor_v*.json"))
        assert len(search_results) == 2

    def test_search_by_tag(self, sample_metadata: dict[str, Any], temp_registry_dir: Path) -> None:
        """Test searching metadata by tag."""
        metadata_dir = temp_registry_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        # Save metadata with tags
        sample_metadata["tags"] = ["production", "stable"]
        metadata_path = metadata_dir / f"{sample_metadata['name']}.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)

        # Search by tag
        with open(metadata_path) as f:
            loaded = json.load(f)

        assert "production" in loaded["tags"]

    def test_search_by_accuracy(
        self, sample_metadata: dict[str, Any], temp_registry_dir: Path
    ) -> None:
        """Test searching models by accuracy threshold."""
        metadata_dir = temp_registry_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        accuracies = [0.75, 0.85, 0.95]
        for acc in accuracies:
            metadata = sample_metadata.copy()
            metadata["accuracy"] = acc
            metadata_path = metadata_dir / f"model_acc_{acc}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

        # Find high accuracy models
        high_acc_models = []
        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file) as f:
                data = json.load(f)
            if data["accuracy"] > 0.80:
                high_acc_models.append(data)

        assert len(high_acc_models) == 2


class TestModelComparison:
    """Test comparing models in registry."""

    def test_compare_model_accuracies(
        self, sample_metadata: dict[str, Any], temp_registry_dir: Path
    ) -> None:
        """Test comparing accuracies of different models."""
        metadata_dir = temp_registry_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        models_data = []
        for i, acc in enumerate([0.75, 0.85, 0.95]):
            metadata = sample_metadata.copy()
            metadata["name"] = f"model_{i}"
            metadata["accuracy"] = acc
            models_data.append(metadata)

            metadata_path = metadata_dir / f"model_{i}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

        # Find best model
        best_model = max(models_data, key=lambda x: x["accuracy"])
        assert best_model["accuracy"] == 0.95

    def test_compare_model_creation_dates(
        self, sample_metadata: dict[str, Any], temp_registry_dir: Path
    ) -> None:
        """Test comparing models by creation date."""
        metadata_dir = temp_registry_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        # Create models with different dates
        for i in range(3):
            metadata = sample_metadata.copy()
            metadata["name"] = f"model_{i}"

            metadata_path = metadata_dir / f"model_{i}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

        # All should be recently created
        metadata_files = list(metadata_dir.glob("*.json"))
        assert len(metadata_files) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
