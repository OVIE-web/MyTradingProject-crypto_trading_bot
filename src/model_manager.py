# src/model_manager.py

import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import os


from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, MODEL_SAVE_PATH, CONFIDENCE_THRESHOLD

# Global variable to store the reverse mapper for signal predictions
reverse_mapper = {0: -1, 1: 0, 2: 1} # Default, will be set during training if needed


def prepare_model_data(df, feature_cols, target_col=TARGET_COLUMN, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Prepare data for model training with proper validation and splitting.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input df must be a pandas DataFrame')
    
    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col} not found in DataFrame')
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Feature columns not found in DataFrame: {missing_cols}')
    
    try:
        X = df[feature_cols]
        y = df[target_col]
        y = y.astype(int) # Ensure target is integer type
        
        # Verify unique classes
        unique_classes = set(y.unique())
        expected_classes = {-1, 0, 1}
        if not unique_classes.issubset(expected_classes):
            logging.warning(f'Invalid target values found: {unique_classes}. Expected: {expected_classes}')
            # Optionally, filter these out or re-map. For now, just warn.
        
        # Stratified split to preserve class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logging.info(f"Data split successfully. Training: {X_train.shape}, Testing: {X_test.shape}")
        logging.info(f"Class distribution before SMOTE (Train):\n{pd.Series(y_train).value_counts()}")
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        logging.info(f"Class distribution after SMOTE (Train):\n{pd.Series(y_train_balanced).value_counts()}")
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    except Exception as e:
        logging.error(f'Error preparing model data: {str(e)}')
        raise


def train_xgboost_model(X_train, y_train, X_test, y_test, random_state=RANDOM_STATE):
    """
    Trains and optimizes an XGBoost model, evaluates it, and saves the best model.
    """
    global reverse_mapper # Declare global to modify it for later use

    try:
        # XGBoost needs labels starting from 0 for multi-class classification
        # Converts [-1, 0, 1] to [0, 1, 2]
        y_train_mapped = y_train + 1 
        y_test_mapped = y_test + 1
        
        # Define reverse mapper specific to this training run's mapping
        reverse_mapper = {0: -1, 1: 0, 2: 1} # This mapping is consistent
        
        # Define parameter grid for optimization
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3, # Corresponds to mapped labels [0, 1, 2]
            random_state=random_state
        )
        
        logging.info("Starting RandomizedSearchCV for XGBoost hyperparameter optimization...")
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=10, # Number of parameter settings that are sampled
            scoring='accuracy',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            random_state=random_state,
            n_jobs=-1, # Use all available cores
            verbose=1
        )
        
        random_search.fit(X_train, y_train_mapped)
        
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        
        # Evaluate on the test set
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Apply confidence threshold for prediction
        y_pred_mapped = np.argmax(y_pred_proba, axis=1)
        max_probs = np.max(y_pred_proba, axis=1)
        
        # Default to 'hold' (mapped value 1) if confidence is below threshold
        y_pred_mapped[max_probs < CONFIDENCE_THRESHOLD] = 1 
        
        y_pred = pd.Series(y_pred_mapped).map(reverse_mapper) # Map back to original labels
        
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f'XGBoost model training completed.')
        logging.info(f'Best parameters: {best_params}')
        logging.info(f'Test accuracy: {accuracy:.4f}')
        logging.info(f'\nClassification Report:\n{classification_report(y_test, y_pred)}')

        # Save the best model
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        best_model.save_model(MODEL_SAVE_PATH)
        logging.info(f"Best XGBoost model saved to {MODEL_SAVE_PATH}")
        
        return best_model, best_params
    except Exception as e:
        logging.error(f'Error training XGBoost model: {str(e)}')
        raise

def load_trained_model(model_path=MODEL_SAVE_PATH):
    """Loads a pre-trained XGBoost model."""
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logging.info(f"XGBoost model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading XGBoost model from {model_path}: {str(e)}")
        raise

def make_predictions(model, X_data, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Generates trading signals from a trained model with a confidence threshold.
    Returns signals mapped back to -1, 0, 1.
    """
    global reverse_mapper # Use the global reverse_mapper

    if not isinstance(model, xgb.XGBClassifier):
        raise TypeError("Model must be an XGBoost Classifier.")
    if X_data.empty:
        logging.warning("Input data for prediction is empty.")
        return pd.Series([], dtype=int) # Return empty series if no data

    try:
        pred_proba = model.predict_proba(X_data)
        predictions_mapped = np.argmax(pred_proba, axis=1)
        max_probs = np.max(pred_proba, axis=1)
        
        # Apply confidence threshold: default to hold if prediction is not confident
        predictions_mapped[max_probs < confidence_threshold] = 1 # 1 maps to 0 (hold)
        
        predictions = pd.Series(predictions_mapped).map(reverse_mapper)
        
        # Align index for consistency
        predictions.index = X_data.index
        
        logging.info(f"Predictions generated with confidence threshold {confidence_threshold}.")
        logging.info(f"Prediction distribution:\n{predictions.value_counts()}")
        
        return predictions, max_probs
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise