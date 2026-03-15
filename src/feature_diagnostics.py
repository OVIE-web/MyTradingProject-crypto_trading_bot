# src/feature_diagnostics.py
"""
Diagnose feature quality and signal generation issues.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ValidationResult(TypedDict):
    """Type definition for feature validation results."""

    valid: bool
    missing_features: list[str]
    non_numeric_features: list[str]
    signal_valid: bool


def normalize_features(
    df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler | None = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize features using StandardScaler.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names to normalize
        scaler: Optional pre-fitted StandardScaler. If None, creates new one.

    Returns:
        Tuple of (normalized DataFrame, fitted scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        df_normalized = df.copy()
        df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df_normalized = df.copy()
        df_normalized[feature_cols] = scaler.transform(df[feature_cols])

    return df_normalized, scaler


def validate_features(
    df: pd.DataFrame, feature_cols: list[str], signal_col: str | None = None
) -> ValidationResult:
    """
    Validate feature columns exist and are numeric.

    Args:
        df: DataFrame to validate
        feature_cols: List of feature column names
        signal_col: Optional signal column to validate

    Returns:
        Dictionary with validation results
    """
    validation: ValidationResult = {
        "valid": True,
        "missing_features": [],
        "non_numeric_features": [],
        "signal_valid": True,
    }

    # Check features exist
    for col in feature_cols:
        if col not in df.columns:
            validation["missing_features"].append(col)
            validation["valid"] = False

    # Check features are numeric
    for col in feature_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            validation["non_numeric_features"].append(col)
            validation["valid"] = False

    # Check signal column if provided
    if signal_col is not None:
        if signal_col not in df.columns:
            validation["signal_valid"] = False
            validation["valid"] = False
        elif not pd.api.types.is_numeric_dtype(df[signal_col]):
            validation["signal_valid"] = False
            validation["valid"] = False

    return validation


class FeatureDiagnostics:
    """Analyze and improve feature quality."""

    @staticmethod
    def analyze_features(df: pd.DataFrame, feature_cols: list[str]) -> dict:
        """
        Analyze feature quality and statistics.

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names

        Returns:
            Dictionary with analysis results
        """
        analysis = {}

        for col in feature_cols:
            if col not in df.columns:
                logger.warning(f"Feature {col} not found in DataFrame")
                continue

            data = df[col].dropna()

            analysis[col] = {
                "count": len(data),
                "missing": df[col].isna().sum(),
                "mean": float(cast(Any, data.mean())),
                "std": float(cast(Any, data.std())),
                "min": float(cast(Any, data.min())),
                "max": float(cast(Any, data.max())),
                "median": float(cast(Any, data.median())),
                "skew": float(cast(Any, data.skew())),
            }

        return analysis

    @staticmethod
    def detect_outliers(df: pd.DataFrame, feature_cols: list[str], threshold: float = 3.0) -> dict:
        """
        Detect outliers using Z-score.

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            threshold: Z-score threshold (default 3.0 = 3 sigma)

        Returns:
            Dictionary with outlier information
        """
        outliers = {}

        for col in feature_cols:
            if col not in df.columns:
                continue

            data = df[col].dropna()
            z_scores = np.abs((data - data.mean()) / data.std())
            outlier_count = (z_scores > threshold).sum()
            outlier_pct = (outlier_count / len(data)) * 100

            outliers[col] = {
                "outlier_count": int(outlier_count),
                "outlier_percentage": float(outlier_pct),
                "threshold": threshold,
            }

        return outliers

    @staticmethod
    def analyze_signal_distribution(df: pd.DataFrame, signal_col: str = "signal") -> dict:
        """
        Analyze signal (target) distribution.

        Args:
            df: DataFrame with signal column
            signal_col: Name of signal column

        Returns:
            Dictionary with signal statistics
        """
        if signal_col not in df.columns:
            raise KeyError(f"Signal column '{signal_col}' not found")

        signal_counts = df[signal_col].value_counts().sort_index()
        total = len(df)

        distribution = {}
        for signal_class, count in signal_counts.items():
            pct = (count / total) * 100
            signal_name = {0: "SELL", 1: "HOLD", 2: "BUY"}.get(
                int(cast(Any, signal_class)), f"Unknown({signal_class})"
            )
            distribution[signal_name] = {
                "count": int(count),
                "percentage": float(pct),
            }

        # Calculate imbalance ratio
        max_count = signal_counts.max()
        min_count = signal_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else 0

        return {
            "distribution": distribution,
            "total_samples": total,
            "imbalance_ratio": float(imbalance_ratio),
            "is_balanced": imbalance_ratio < 2.0,  # Good if < 2x difference
        }

    @staticmethod
    def check_feature_correlation(
        df: pd.DataFrame, feature_cols: list[str], signal_col: str = "signal"
    ) -> dict:
        """
        Check correlation between features and signal.

        Args:
            df: DataFrame with features and signal
            feature_cols: List of feature column names
            signal_col: Name of signal column

        Returns:
            Dictionary with correlation information
        """
        if signal_col not in df.columns:
            raise KeyError(f"Signal column '{signal_col}' not found")

        df_numeric = df[feature_cols + [signal_col]].select_dtypes(include=[np.number])
        corr_matrix = df_numeric.corr()

        # Get correlation with signal
        signal_corr = corr_matrix[signal_col].drop(signal_col).sort_values(ascending=False)

        correlation_info = {
            "feature_signal_correlation": signal_corr.to_dict(),
            "strong_correlations": float((abs(signal_corr) > 0.3).sum()),  # Corr > 0.3
            "weak_correlations": float((abs(signal_corr) < 0.1).sum()),  # Corr < 0.1
        }

        return correlation_info

    @staticmethod
    def suggest_improvements(
        analysis: dict, outliers: dict, signal_dist: dict, correlation: dict
    ) -> list[str]:
        """
        Generate improvement suggestions based on diagnostics.

        Args:
            analysis: Feature analysis results
            outliers: Outlier detection results
            signal_dist: Signal distribution results
            correlation: Correlation analysis results

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Check for missing values
        for feature, stats in analysis.items():
            if stats["missing"] > 0:
                pct = (stats["missing"] / (stats["missing"] + stats["count"])) * 100
                suggestions.append(
                    f"⚠️ Feature '{feature}' has {pct:.1f}% missing values - consider imputation"
                )

        # Check for outliers
        for feature, outlier_info in outliers.items():
            if outlier_info["outlier_percentage"] > 5:
                suggestions.append(
                    f"⚠️ Feature '{feature}' has {outlier_info['outlier_percentage']:.1f}% outliers - "
                    f"consider clipping or robust scaling"
                )

        # Check signal balance
        if not signal_dist.get("is_balanced", False):
            ratio = signal_dist.get("imbalance_ratio", 0)
            suggestions.append(
                f"⚠️ Signal distribution imbalanced (ratio={ratio:.2f}x) - "
                f"consider class weighting in model training"
            )

        # Check feature-signal correlation
        strong_corr = correlation.get("strong_correlations", 0)
        if strong_corr < 3:
            suggestions.append(
                f"⚠️ Only {strong_corr} features strongly correlate with signal - "
                f"consider feature engineering (new indicators, lagged features)"
            )

        if len(suggestions) == 0:
            suggestions.append(
                "✅ Data looks good! Model improvement likely needs hyperparameter tuning or more data."
            )

        return suggestions


def run_diagnostics(csv_path: str, feature_cols: list[str], signal_col: str = "signal") -> None:
    """
    Run complete feature diagnostics.

    Args:
        csv_path: Path to CSV file
        feature_cols: List of feature column names
        signal_col: Name of signal column
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load data
    logger.info(f"📂 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Run diagnostics
    logger.info("\n" + "=" * 70)
    logger.info("📊 FEATURE DIAGNOSTICS")
    logger.info("=" * 70)

    diagnostics = FeatureDiagnostics()

    # Feature analysis
    logger.info("\n1️⃣  FEATURE STATISTICS")
    logger.info("-" * 70)
    analysis = diagnostics.analyze_features(df, feature_cols)
    for feature, stats in analysis.items():
        logger.info(
            f"{feature:20s} | Mean={stats['mean']:8.3f} | Std={stats['std']:8.3f} | "
            f"Min={stats['min']:8.3f} | Max={stats['max']:8.3f} | Missing={stats['missing']}"
        )

    # Outlier detection
    logger.info("\n2️⃣  OUTLIER DETECTION")
    logger.info("-" * 70)
    outliers = diagnostics.detect_outliers(df, feature_cols)
    for feature, info in outliers.items():
        logger.info(
            f"{feature:20s} | Outliers={info['outlier_count']:4d} ({info['outlier_percentage']:5.2f}%)"
        )

    # Signal distribution
    logger.info("\n3️⃣  SIGNAL DISTRIBUTION")
    logger.info("-" * 70)
    signal_dist = diagnostics.analyze_signal_distribution(df, signal_col)
    logger.info(f"Total samples: {signal_dist['total_samples']}")
    for signal_name, info in signal_dist["distribution"].items():
        logger.info(f"  {signal_name:6s}: {info['count']:4d} ({info['percentage']:5.2f}%)")
    logger.info(
        f"Imbalance ratio: {signal_dist['imbalance_ratio']:.2f}x {'❌ IMBALANCED' if not signal_dist['is_balanced'] else '✅ BALANCED'}"
    )

    # Feature-signal correlation
    logger.info("\n4️⃣  FEATURE-SIGNAL CORRELATION")
    logger.info("-" * 70)
    correlation = diagnostics.check_feature_correlation(df, feature_cols, signal_col)
    for feature, corr_value in correlation["feature_signal_correlation"].items():
        strength = (
            "STRONG" if abs(corr_value) > 0.3 else "WEAK" if abs(corr_value) < 0.1 else "MODERATE"
        )
        logger.info(f"{feature:20s} | Correlation={corr_value:7.4f} ({strength})")

    # Suggestions
    logger.info("\n5️⃣  IMPROVEMENT SUGGESTIONS")
    logger.info("-" * 70)
    suggestions = diagnostics.suggest_improvements(analysis, outliers, signal_dist, correlation)
    for i, suggestion in enumerate(suggestions, 1):
        logger.info(f"{i}. {suggestion}")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.error("Usage: python feature_diagnostics.py <csv_path> [feature_cols...]")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Default feature columns
    from src.config import FEATURE_COLUMNS

    feature_cols = sys.argv[2:] if len(sys.argv) > 2 else FEATURE_COLUMNS

    run_diagnostics(csv_path, feature_cols)
