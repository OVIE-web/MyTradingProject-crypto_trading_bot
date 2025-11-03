import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.utils import class_weight
from joblib import dump

# -----------------------------------
# Config
# -----------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "src/models/xgboost_model.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------
# Data Preparation
# -----------------------------------
def prepare_model_data(df: pd.DataFrame, feature_cols: list, target_col: str):
    try:
        X = df[feature_cols]
        y = df[target_col]

        # Stratified split for better label balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        logger.info(f"Data split successfully. Training: {X_train.shape}, Testing: {X_test.shape}")
        logger.info(f"Class distribution before SMOTE (Train):\n{y_train.value_counts()}")

        # SMOTE with dynamic neighbors
        min_class_size = y_train.value_counts().min()
        k_neighbors = max(1, min(5, min_class_size - 1))
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)

        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        logger.info(f"Class distribution after SMOTE (Train):\n{y_train_balanced.value_counts()}")

        return X_train_balanced, X_test, y_train_balanced, y_test

    except Exception as e:
        logger.error(f"Error preparing model data: {e}", exc_info=True)
        raise


# -----------------------------------
# Model Training
# -----------------------------------
def train_xgboost_model(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE):
    try:
        # Compute class weights
        weights = class_weight.compute_sample_weight("balanced", y_train)
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            use_label_encoder=False
        )

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "gamma": [0, 1]
        }

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=3,
            scoring="accuracy",
            cv=2,
            verbose=0,
            random_state=random_state,
            n_jobs=-1
        )
        search.fit(X_train, y_train, sample_weight=weights)

        best_model = search.best_estimator_
        best_params = search.best_params_

        # Evaluate
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"✅ Test accuracy: {acc:.4f}")
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

        # Save model (mockable)
        model_path = MODEL_SAVE_PATH
        if hasattr(best_model, "save_model"):
            best_model.save_model(model_path)
        else:
            dump(best_model, model_path)

        logger.info(f"Model saved to {model_path}")
        return best_model, best_params

    except Exception as e:
        logger.error(f"Error training XGBoost model: {e}", exc_info=True)
        raise


# -----------------------------------
# Predictions
# -----------------------------------
def make_predictions(model, X_data, confidence_threshold=0.3):
    try:
        probabilities = model.predict_proba(X_data)
        confidence_scores = np.max(probabilities, axis=1)
        preds = np.argmax(probabilities, axis=1) - 1  # shift classes 0->-1, 1->0, 2->1

        preds_confident = np.where(confidence_scores >= confidence_threshold, preds, 0)
        predictions = pd.Series(preds_confident, name="signal")

        logger.info(f"Predictions generated with confidence threshold {confidence_threshold}.")
        logger.info(f"Prediction distribution:\n{predictions.value_counts()}")

        return predictions, pd.Series(confidence_scores, name="confidence")

    except Exception as e:
        logger.error(f"Error making predictions: {e}", exc_info=True)
        raise
