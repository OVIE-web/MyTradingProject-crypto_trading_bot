"""Test for feature diagnostics module"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.feature_diagnostics import (
    FeatureDiagnostics,
    normalize_features,
    validate_features,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing diagnostics."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.uniform(0, 10, n_samples),
            "signal": np.random.choice([0, 1, 2], n_samples),
        }
    )


def test_normalize_features(sample_data: pd.DataFrame) -> None:
    """Test feature normalization logic."""
    cols = ["feature1", "feature2"]
    df_norm, scaler = normalize_features(sample_data.copy(), cols)

    assert isinstance(df_norm, pd.DataFrame)
    assert isinstance(scaler, StandardScaler)
    assert abs(df_norm["feature1"].mean()) < 1e-10
    assert abs(df_norm["feature1"].std(ddof=0) - 1.0) < 1e-10

    # Test with pre-fitted scaler
    df_norm2, _ = normalize_features(sample_data.copy(), cols, scaler=scaler)
    pd.testing.assert_frame_equal(df_norm, df_norm2)


def test_validate_features(sample_data: pd.DataFrame) -> None:
    """Test feature validation logic."""
    cols = ["feature1", "feature2"]

    # Valid case
    v1 = validate_features(sample_data, cols, signal_col="signal")
    assert v1["valid"] is True
    assert len(v1["missing_features"]) == 0
    assert len(v1["non_numeric_features"]) == 0
    assert v1["signal_valid"] is True

    # Missing feature
    v2 = validate_features(sample_data, ["missing_col"])
    assert v2["valid"] is False
    assert "missing_col" in v2["missing_features"]

    # Non-numeric feature
    df_bad = sample_data.copy()
    df_bad["bad_col"] = "not a number"
    v3 = validate_features(df_bad, ["bad_col"])
    assert v3["valid"] is False
    assert "bad_col" in v3["non_numeric_features"]


def test_analyze_features(sample_data: pd.DataFrame) -> None:
    """Test feature statistics analysis."""
    diagnose = FeatureDiagnostics()
    cols = ["feature1", "feature2"]
    analysis = diagnose.analyze_features(sample_data, cols)

    assert "feature1" in analysis
    assert "feature2" in analysis
    assert analysis["feature1"]["count"] == 100
    assert "mean" in analysis["feature1"]
    assert "std" in analysis["feature1"]


def test_detect_outliers(sample_data: pd.DataFrame) -> None:
    """Test outlier detection logic."""
    # Add an outlier
    df_outlier = sample_data.copy()
    df_outlier.loc[0, "feature1"] = 100.0

    diagnose = FeatureDiagnostics()
    outliers = diagnose.detect_outliers(df_outlier, ["feature1"])

    assert outliers["feature1"]["outlier_count"] >= 1
    assert outliers["feature1"]["outlier_percentage"] > 0


def test_analyze_signal_distribution(sample_data: pd.DataFrame) -> None:
    """Test signal distribution analysis."""
    diagnose = FeatureDiagnostics()
    dist = diagnose.analyze_signal_distribution(sample_data, "signal")

    assert dist["total_samples"] == 100
    assert "BUY" in dist["distribution"]
    assert "SELL" in dist["distribution"]
    assert "HOLD" in dist["distribution"]
    assert "is_balanced" in dist


def test_check_feature_correlation(sample_data: pd.DataFrame) -> None:
    """Test feature-signal correlation check."""
    diagnose = FeatureDiagnostics()
    cols = ["feature1", "feature2"]
    corr = diagnose.check_feature_correlation(sample_data, cols, "signal")

    assert "feature_signal_correlation" in corr
    assert "feature1" in corr["feature_signal_correlation"]


def test_suggest_improvements() -> None:
    """Test improvement suggestions logic."""
    diagnose = FeatureDiagnostics()

    analysis = {"f1": {"missing": 10, "count": 90}}
    outliers = {"f1": {"outlier_percentage": 10}}
    signal_dist = {"is_balanced": False, "imbalance_ratio": 3.0}
    correlation = {"strong_correlations": 1}

    suggestions = diagnose.suggest_improvements(analysis, outliers, signal_dist, correlation)

    assert len(suggestions) > 0
    assert any("missing" in s for s in suggestions)
    assert any("outliers" in s for s in suggestions)
    assert any("imbalanced" in s for s in suggestions)
    assert any(
        "not 3 features strongly correlate" in s or "Only 1 features" in s for s in suggestions
    )
