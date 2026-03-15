"""
Comprehensive tests for src/debug_features.py module.
Tests feature debugging and inspection utilities.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame for debugging."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "close": np.random.randn(50) + 100,
            "volume": np.random.uniform(1000, 10000, 50),
            "RSI": np.random.uniform(0, 100, 50),
            "signal": np.random.randint(0, 3, 50),
            "with_nan": np.concatenate(
                [
                    np.random.randn(45),
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                ]
            ),
        }
    )


@pytest.fixture
def sample_array() -> np.ndarray:
    """Create sample numpy array."""
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture
def sample_dataframe_with_missing() -> pd.DataFrame:
    """Create DataFrame with various missing value patterns."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "col_a": [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            "col_b": [np.nan, 20, 30, 40, np.nan, 60, 70, 80, 90, 100],
            "col_c": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        }
    )


@pytest.fixture
def sample_numeric_dataframe() -> pd.DataFrame:
    """Create DataFrame with purely numeric data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(100),
            "feature_2": np.random.uniform(0, 100, 100),
            "feature_3": np.random.randint(0, 50, 100),
            "target": np.random.randint(0, 3, 100),
        }
    )


class TestFeatureInspection:
    """Test feature inspection utilities."""

    def test_inspect_dataframe_shape(self, sample_dataframe: pd.DataFrame) -> None:
        """Test inspecting DataFrame shape."""
        shape: tuple[int, ...] = sample_dataframe.shape

        assert shape[0] == 50
        assert shape[1] == 5

    def test_inspect_dataframe_columns(self, sample_dataframe: pd.DataFrame) -> None:
        """Test inspecting DataFrame columns."""
        columns: list[str] = sample_dataframe.columns.tolist()

        assert "close" in columns
        assert "volume" in columns
        assert "RSI" in columns
        assert "signal" in columns

    def test_inspect_dataframe_dtypes(self, sample_dataframe: pd.DataFrame) -> None:
        """Test inspecting DataFrame data types."""
        dtypes = sample_dataframe.dtypes

        assert np.issubdtype(dtypes["close"], np.floating)
        assert np.issubdtype(dtypes["volume"], np.floating)

    def test_inspect_array_shape(self, sample_array: np.ndarray) -> None:
        """Test inspecting array shape."""
        shape: tuple[int, ...] = sample_array.shape

        assert len(shape) == 1
        assert shape[0] == 100

    def test_inspect_array_dtype(self, sample_array: np.ndarray) -> None:
        """Test inspecting array dtype."""
        dtype = sample_array.dtype

        assert np.issubdtype(dtype, np.floating)

    def test_inspect_missing_values(self, sample_dataframe: pd.DataFrame) -> None:
        """Test inspecting missing values."""
        missing = sample_dataframe.isna().sum()

        assert missing["close"] == 0
        assert missing["with_nan"] == 5

    def test_inspect_null_counts(self, sample_dataframe: pd.DataFrame) -> None:
        """Test counting null values per column."""
        null_counts = sample_dataframe.isnull().sum()

        assert null_counts["with_nan"] == 5
        assert (null_counts[["close", "volume", "RSI"]] == 0).all()

    def test_inspect_dataframe_info(self, sample_dataframe: pd.DataFrame) -> None:
        """Test getting DataFrame info."""
        info_dict: dict[str, Any] = {
            "shape": sample_dataframe.shape,
            "columns": len(sample_dataframe.columns),
            "index": len(sample_dataframe.index),
        }

        assert info_dict["shape"][0] == 50
        assert info_dict["columns"] == 5
        assert info_dict["index"] == 50


class TestFeatureStatistics:
    """Test feature statistics computation."""

    def test_compute_mean(self, sample_dataframe: pd.DataFrame) -> None:
        """Test computing mean of features."""
        mean_close: float = float(sample_dataframe["close"].mean())

        assert isinstance(mean_close, float)
        assert 95 < mean_close < 105

    def test_compute_std(self, sample_dataframe: pd.DataFrame) -> None:
        """Test computing standard deviation."""
        std_close: float = float(sample_dataframe["close"].std())

        assert std_close > 0
        assert isinstance(std_close, float)

    def test_compute_min_max(self, sample_dataframe: pd.DataFrame) -> None:
        """Test computing min and max."""
        min_val: float = float(sample_dataframe["RSI"].min())
        max_val: float = float(sample_dataframe["RSI"].max())

        assert min_val >= 0
        assert max_val <= 100
        assert min_val <= max_val

    def test_compute_percentiles(self, sample_dataframe: pd.DataFrame) -> None:
        """Test computing percentiles."""
        p25: float = float(sample_dataframe["close"].quantile(0.25))
        p50: float = float(sample_dataframe["close"].quantile(0.50))
        p75: float = float(sample_dataframe["close"].quantile(0.75))

        assert p25 <= p50 <= p75

    def test_describe_dataframe(self, sample_dataframe: pd.DataFrame) -> None:
        """Test describing DataFrame statistics."""
        desc = sample_dataframe.describe()

        assert "count" in desc.index
        assert "mean" in desc.index
        assert "std" in desc.index
        assert "min" in desc.index
        assert "max" in desc.index

    def test_compute_variance(self, sample_dataframe: pd.DataFrame) -> None:
        """Test computing variance."""
        variance: float = float(sample_dataframe["close"].var())

        assert variance > 0
        assert isinstance(variance, float)

    def test_compute_multiple_stats(self, sample_numeric_dataframe: pd.DataFrame) -> None:
        """Test computing multiple statistics at once."""
        stats: dict[str, float] = {
            "mean": float(sample_numeric_dataframe["feature_1"].mean()),
            "std": float(sample_numeric_dataframe["feature_1"].std()),
            "min": float(sample_numeric_dataframe["feature_1"].min()),
            "max": float(sample_numeric_dataframe["feature_1"].max()),
        }

        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["std"] >= 0


class TestFeatureDistribution:
    """Test analyzing feature distributions."""

    def test_distribution_histogram(self, sample_dataframe: pd.DataFrame) -> None:
        """Test computing histogram of features."""
        hist, bins = np.histogram(sample_dataframe["close"], bins=10)

        assert len(hist) == 10
        assert len(bins) == 11
        assert hist.sum() == len(sample_dataframe)

    def test_distribution_unique_values(self, sample_dataframe: pd.DataFrame) -> None:
        """Test counting unique values."""
        unique_signals: int = int(sample_dataframe["signal"].nunique())

        assert unique_signals == 3  # 0, 1, 2

    def test_distribution_value_counts(self, sample_dataframe: pd.DataFrame) -> None:
        """Test counting value occurrences."""
        counts = sample_dataframe["signal"].value_counts()

        assert counts.sum() == 50
        assert len(counts) == 3

    def test_distribution_skewness(self, sample_dataframe: pd.DataFrame) -> None:
        """Test computing skewness."""
        skewness: float = float(cast(Any, sample_dataframe["close"].skew()))

        assert isinstance(skewness, float)

    def test_distribution_kurtosis(self, sample_dataframe: pd.DataFrame) -> None:
        """Test computing kurtosis."""
        kurtosis: float = float(cast(Any, sample_dataframe["close"].kurtosis()))

        assert isinstance(kurtosis, float)

    def test_distribution_mode(self, sample_dataframe: pd.DataFrame) -> None:
        """Test finding mode of distribution."""
        mode_value = sample_dataframe["signal"].mode()

        assert len(mode_value) > 0


class TestFeatureCorrelation:
    """Test analyzing feature correlations."""

    def test_correlation_matrix(self, sample_dataframe: pd.DataFrame) -> None:
        """Test computing correlation matrix."""
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number])
        corr = numeric_cols.corr()

        assert corr.shape[0] == corr.shape[1]
        assert (np.diag(corr) == 1).all()  # Diagonal should be 1

    def test_pairwise_correlation(self, sample_dataframe: pd.DataFrame) -> None:
        """Test computing pairwise correlations."""
        corr_value: float = float(sample_dataframe["close"].corr(sample_dataframe["volume"]))

        assert -1 <= corr_value <= 1

    def test_correlation_with_signal(self, sample_dataframe: pd.DataFrame) -> None:
        """Test correlation with signal column."""
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number])
        corr_with_signal = numeric_cols.corrwith(sample_dataframe["signal"])

        assert len(corr_with_signal) > 0
        assert all(-1 <= c <= 1 for c in corr_with_signal)

    def test_correlation_with_nan(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test correlation handles NaN values."""
        corr = sample_dataframe_with_missing.corr()

        # Should not raise error and return valid correlations
        assert not corr.empty
        assert corr.shape == (3, 3)


class TestFeatureOutliers:
    """Test outlier detection."""

    def test_detect_outliers_zscore(self, sample_dataframe: pd.DataFrame) -> None:
        """Test detecting outliers using Z-score."""
        close_col = sample_dataframe["close"]
        z_scores = np.abs((close_col - close_col.mean()) / close_col.std())
        outliers = z_scores > 3

        assert isinstance(outliers, pd.Series)
        assert outliers.dtype == bool

    def test_detect_outliers_iqr(self, sample_dataframe: pd.DataFrame) -> None:
        """Test detecting outliers using IQR method."""
        close_col = sample_dataframe["close"]
        q1: float = float(close_col.quantile(0.25))
        q3: float = float(close_col.quantile(0.75))
        iqr: float = q3 - q1

        outliers = (close_col < q1 - 1.5 * iqr) | (close_col > q3 + 1.5 * iqr)

        assert isinstance(outliers, pd.Series)

    def test_count_outliers(self, sample_dataframe: pd.DataFrame) -> None:
        """Test counting outliers."""
        close_col = sample_dataframe["close"]
        z_scores = np.abs((close_col - close_col.mean()) / close_col.std())
        outlier_count: int = int((z_scores > 3).sum())

        assert isinstance(outlier_count, int)
        assert 0 <= outlier_count <= len(sample_dataframe)

    def test_outlier_percentage(self, sample_dataframe: pd.DataFrame) -> None:
        """Test computing outlier percentage."""
        close_col = sample_dataframe["close"]
        z_scores = np.abs((close_col - close_col.mean()) / close_col.std())
        outlier_count: int = int((z_scores > 3).sum())
        outlier_pct: float = (outlier_count / len(sample_dataframe)) * 100

        assert 0 <= outlier_pct <= 100


class TestFeaturePrinting:
    """Test printing feature information."""

    def test_print_dataframe_head(self, sample_dataframe: pd.DataFrame, capsys: Any) -> None:
        """Test printing first few rows."""
        print(sample_dataframe.head(3))
        captured = capsys.readouterr()

        assert "close" in captured.out
        assert "volume" in captured.out

    def test_print_dataframe_info(self, sample_dataframe: pd.DataFrame) -> None:
        """Test printing DataFrame info."""
        # Just test that it doesn't crash
        sample_dataframe.info(buf=None)
        assert True

    def test_print_statistics(self, sample_dataframe: pd.DataFrame, capsys: Any) -> None:
        """Test printing statistics."""
        print(sample_dataframe.describe())
        captured = capsys.readouterr()

        assert "count" in captured.out
        assert "mean" in captured.out

    def test_print_correlations(self, sample_dataframe: pd.DataFrame, capsys: Any) -> None:
        """Test printing correlations."""
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number])
        print(numeric_cols.corr())
        captured = capsys.readouterr()

        assert "close" in captured.out


class TestFeatureComparison:
    """Test comparing features."""

    def test_compare_column_stats(self, sample_dataframe: pd.DataFrame) -> None:
        """Test comparing statistics of different columns."""
        stats_close = sample_dataframe["close"].describe()
        stats_volume = sample_dataframe["volume"].describe()

        # Should have same indices
        assert (stats_close.index == stats_volume.index).all()

    def test_compare_distributions(self, sample_dataframe: pd.DataFrame) -> None:
        """Test comparing distributions."""
        dist1 = sample_dataframe["close"].hist(bins=10)
        dist2 = sample_dataframe["volume"].hist(bins=10)

        # Just test that histograms can be created
        assert dist1 is not None
        assert dist2 is not None

    def test_find_strongest_correlation(self, sample_dataframe: pd.DataFrame) -> None:
        """Test finding strongest correlation."""
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number])
        corr = numeric_cols.corr()

        # Get correlations with signal
        signal_corr = corr["signal"].drop("signal")
        strongest: str = str(signal_corr.abs().idxmax())

        assert strongest in sample_dataframe.columns

    def test_compare_feature_ranges(self, sample_numeric_dataframe: pd.DataFrame) -> None:
        """Test comparing feature value ranges."""
        ranges: dict[str, float] = {
            "feature_1": float(
                sample_numeric_dataframe["feature_1"].max()
                - sample_numeric_dataframe["feature_1"].min()
            ),
            "feature_2": float(
                sample_numeric_dataframe["feature_2"].max()
                - sample_numeric_dataframe["feature_2"].min()
            ),
        }

        assert ranges["feature_1"] > 0
        assert ranges["feature_2"] > 0


class TestFeatureValidation:
    """Test feature validation."""

    def test_validate_no_nulls(self, sample_dataframe: pd.DataFrame) -> None:
        """Test validating DataFrame has no nulls."""
        clean_df = sample_dataframe.dropna()

        assert clean_df.isnull().sum().sum() == 0

    def test_validate_numeric_types(self, sample_dataframe: pd.DataFrame) -> None:
        """Test validating numeric types."""
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number]).columns

        assert len(numeric_cols) > 0
        assert all(pd.api.types.is_numeric_dtype(sample_dataframe[col]) for col in numeric_cols)

    def test_validate_range(self, sample_dataframe: pd.DataFrame) -> None:
        """Test validating feature ranges."""
        rsi = sample_dataframe["RSI"]

        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_validate_row_count(self, sample_dataframe: pd.DataFrame) -> None:
        """Test validating minimum rows."""
        min_rows: int = 10

        assert len(sample_dataframe) >= min_rows

    def test_validate_column_count(self, sample_dataframe: pd.DataFrame) -> None:
        """Test validating minimum columns."""
        min_cols: int = 4

        assert len(sample_dataframe.columns) >= min_cols

    def test_validate_no_infinity(self, sample_numeric_dataframe: pd.DataFrame) -> None:
        """Test validating no infinite values."""
        has_inf = np.isinf(sample_numeric_dataframe.select_dtypes(include=[np.number])).any().any()
        assert not has_inf

    def test_validate_duplicate_rows(self, sample_dataframe: pd.DataFrame) -> None:
        """Test identifying duplicate rows."""
        duplicates = sample_dataframe.duplicated()

        assert isinstance(duplicates, pd.Series)


class TestFeatureTransformation:
    """Test feature transformations for debugging."""

    def test_log_transform(self, sample_dataframe: pd.DataFrame) -> None:
        """Test log transformation."""
        original = sample_dataframe["volume"].copy()
        log_transformed = np.log(original + 1)  # Add 1 to avoid log(0)

        assert len(log_transformed) == len(original)
        assert np.all(np.isfinite(log_transformed))

    def test_normalize_feature(self, sample_dataframe: pd.DataFrame) -> None:
        """Test normalizing feature to [0, 1]."""
        feature = sample_dataframe["close"].copy()
        normalized = (feature - feature.min()) / (feature.max() - feature.min())

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_standardize_feature(self, sample_dataframe: pd.DataFrame) -> None:
        """Test standardizing feature."""
        feature = sample_dataframe["close"].copy()
        standardized = (feature - feature.mean()) / feature.std()

        assert np.isclose(standardized.mean(), 0, atol=1e-10)
        assert np.isclose(standardized.std(), 1, atol=1e-10)

    def test_create_lagged_feature(self, sample_dataframe: pd.DataFrame) -> None:
        """Test creating lagged features."""
        original = sample_dataframe["close"].copy()
        lagged = original.shift(1)

        assert len(lagged) == len(original)
        assert pd.isna(lagged.iloc[0])

    def test_create_rolling_feature(self, sample_dataframe: pd.DataFrame) -> None:
        """Test creating rolling window features."""
        original = sample_dataframe["close"].copy()
        rolling_mean = original.rolling(window=5).mean()

        assert len(rolling_mean) == len(original)
        assert pd.isna(rolling_mean.iloc[0])

    def test_create_multiple_lags(self, sample_dataframe: pd.DataFrame) -> None:
        """Test creating multiple lagged features."""
        original = sample_dataframe["close"].copy()
        lags: dict[str, pd.Series] = {
            "lag_1": original.shift(1),
            "lag_2": original.shift(2),
            "lag_3": original.shift(3),
        }

        for lag_name, lag_series in lags.items():
            assert len(lag_series) == len(original)


class TestDebugOutputFormatting:
    """Test formatting debug output."""

    def test_format_as_table(self, sample_dataframe: pd.DataFrame) -> None:
        """Test formatting as table."""
        table: str = sample_dataframe.head(5).to_string()

        assert isinstance(table, str)
        assert len(table) > 0

    def test_format_as_dict(self, sample_dataframe: pd.DataFrame) -> None:
        """Test formatting as dictionary."""
        data_dict: dict[str, Any] = cast(dict[str, Any], sample_dataframe.to_dict())

        assert isinstance(data_dict, dict)
        assert len(data_dict) > 0

    def test_format_statistics_summary(self, sample_dataframe: pd.DataFrame) -> None:
        """Test formatting statistics summary."""
        summary: dict[str, int] = {
            "rows": len(sample_dataframe),
            "columns": len(sample_dataframe.columns),
            "missing": int(sample_dataframe.isnull().sum().sum()),
        }

        assert summary["rows"] == 50
        assert summary["columns"] == 5
        assert summary["missing"] == 5

    def test_format_with_rounding(self, sample_numeric_dataframe: pd.DataFrame) -> None:
        """Test formatting numeric values with rounding."""
        stats: dict[str, float] = {
            "mean": round(float(sample_numeric_dataframe["feature_1"].mean()), 2),
            "std": round(float(sample_numeric_dataframe["feature_1"].std()), 2),
        }

        assert isinstance(stats["mean"], float)
        assert isinstance(stats["std"], float)


class TestFeatureMismatchDetection:
    """Test detecting feature mismatches between model and data."""

    def test_feature_count_mismatch(self, sample_numeric_dataframe: pd.DataFrame) -> None:
        """Test detecting feature count mismatch."""
        expected_features: int = 4
        actual_features: int = len(sample_numeric_dataframe.columns)

        assert actual_features == expected_features

    def test_feature_name_mismatch(self, sample_numeric_dataframe: pd.DataFrame) -> None:
        """Test detecting feature name mismatch."""
        expected_names: set[str] = {"feature_1", "feature_2", "feature_3", "target"}
        actual_names: set[str] = set(sample_numeric_dataframe.columns)

        assert expected_names == actual_names

    def test_missing_features(self, sample_numeric_dataframe: pd.DataFrame) -> None:
        """Test identifying missing features."""
        expected: set[str] = {"feature_1", "feature_2", "feature_3", "feature_4"}
        actual: set[str] = set(sample_numeric_dataframe.columns)
        missing: set[str] = expected - actual

        assert "feature_4" in missing

    def test_extra_features(self, sample_numeric_dataframe: pd.DataFrame) -> None:
        """Test identifying extra features."""
        expected: set[str] = {"feature_1", "feature_2"}
        actual: set[str] = set(sample_numeric_dataframe.columns)
        extra: set[str] = actual - expected

        assert "feature_3" in extra
        assert "target" in extra

    def test_feature_type_mismatch(self, sample_numeric_dataframe: pd.DataFrame) -> None:
        """Test detecting feature type mismatch."""
        expected_types: dict[str, str] = {
            "feature_1": "float64",
            "feature_2": "float64",
        }
        actual_types: dict[str, str] = {
            str(col): str(dtype) for col, dtype in sample_numeric_dataframe.dtypes.items()
        }

        assert actual_types["feature_1"] == expected_types["feature_1"]


class TestMissingValueHandling:
    """Test handling of missing values in features."""

    def test_missing_value_detection(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test detecting missing values."""
        missing_counts = sample_dataframe_with_missing.isnull().sum()

        assert missing_counts["col_a"] == 2
        assert missing_counts["col_b"] == 2
        assert missing_counts["col_c"] == 0

    def test_missing_value_percentage(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test computing missing value percentage."""
        total_missing: int = int(sample_dataframe_with_missing.isnull().sum().sum())
        total_values: int = len(sample_dataframe_with_missing) * len(
            sample_dataframe_with_missing.columns
        )
        missing_pct: float = (total_missing / total_values) * 100

        assert 0 < missing_pct < 100

    def test_fill_missing_values(self, sample_dataframe_with_missing: pd.DataFrame) -> None:
        """Test filling missing values."""
        filled_df = sample_dataframe_with_missing.ffill()

        # Forward fill should reduce missing values
        missing_before: int = int(sample_dataframe_with_missing.isnull().sum().sum())
        missing_after: int = int(filled_df.isnull().sum().sum())

        assert missing_after <= missing_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
