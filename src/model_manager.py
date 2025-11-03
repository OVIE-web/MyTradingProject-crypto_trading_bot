import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb

MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "src/models/xgboost_model.json")

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

def prepare_model_data(df, feature_cols, target_col):
    try:
        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        logger.info(f"Data split successfully. Training: {X_train.shape}, Testing: {X_test.shape}")
        logger.info("Class distribution before SMOTE (Train):\n%s", y_train.value_counts())

        smote_k = min(5, y_train.value_counts().min() - 1)
        smote_k = max(1, smote_k)
        smote = SMOTE(k_neighbors=smote_k, random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        logger.info("Class distribution after SMOTE (Train):\n%s", y_train_balanced.value_counts())
        return X_train_balanced, X_test, y_train_balanced, y_test
    except Exception as e:
        logger.error("Error preparing model data: %s", e, exc_info=True)
        raise


def train_xgboost_model(
    X_train, y_train, X_test, y_test, random_state: int = 42, model_path: str | None = None
):
    try:
        # Encode target labels
        y_train_encoded = y_train.replace({-1: 0, 0: 1, 1: 2})
        y_test_encoded = y_test.replace({-1: 0, 0: 1, 1: 2})

        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1,
        )

        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        }

        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=3,
            cv=2,
            scoring="accuracy",
            random_state=random_state,
            verbose=0,
            n_jobs=-1,
        )

        weights = np.ones(len(y_train_encoded))
        search.fit(X_train, y_train_encoded, sample_weight=weights)

        best_model = search.best_estimator_
        best_params = search.best_params_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test_encoded, y_pred)
        report = classification_report(y_test_encoded, y_pred)

        logger.info(f"✅ Test accuracy: {acc:.4f}")
        logger.info("Classification Report:\n%s", report)

        # ✅ Use provided path or fallback to constant
        model_path = model_path or MODEL_SAVE_PATH
        dir_name = os.path.dirname(model_path)
        if dir_name:  # only create if non-empty
            os.makedirs(dir_name, exist_ok=True)

        best_model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        return best_model, best_params

    except Exception as e:
        logger.error("Error training XGBoost model: %s", e, exc_info=True)
        raise
    
def load_trained_model(model_path: str | None = None):
    model_path = model_path or MODEL_SAVE_PATH
    model = xgb.XGBClassifier()
    if not os.path.exists(model_path):
        logger.warning(f"No trained model found at {model_path}, returning untrained model")
        return model
    model.load_model(model_path)
    return model



def make_predictions(model, X_data, confidence_threshold=CONFIDENCE_THRESHOLD):
    try:
        if X_data.empty:
            logger.warning("Empty input received for prediction.")
            return pd.Series(dtype=int), pd.Series(dtype=float)

        proba = model.predict_proba(X_data)
        max_prob = np.max(proba, axis=1)
        preds = np.argmax(proba, axis=1)

        # ✅ Decode predictions back to [-1, 0, 1]
        pred_map = {0: -1, 1: 0, 2: 1}
        decoded_preds = pd.Series([pred_map[p] for p in preds], name="signal")

        decoded_preds[max_prob < confidence_threshold] = 0  # Hold if confidence is low
        confidence = pd.Series(max_prob, name="confidence")

        logger.info(f"Predictions generated with confidence threshold {confidence_threshold}.")
        logger.info("Prediction distribution:\n%s", decoded_preds.value_counts())

        return decoded_preds, confidence

    except Exception as e:
        logger.error("Error making predictions: %s", e, exc_info=True)
        return pd.Series(dtype=int), pd.Series(dtype=float)
