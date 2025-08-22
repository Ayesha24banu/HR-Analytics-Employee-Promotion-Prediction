"""
Model Training Script for HR Analytics
--------------------------------------
Business Context:
    We want to predict employee promotions using historical HR data.
    This script trains multiple models, tunes them, evaluates performance, and saves the best one.

steps:
     1. Loads preprocessed training data
     2. Splits into training & test sets
     3. Trains multiple baseline models
     4. Tunes hyperparameters for top models
     5. Evaluates final models
     6. Saves the best model for deployment
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_and_save_model(train_path: str):
    """
    Train multiple models, tune hyperparameters, evaluate on test set,
    and save the best model.

    Args:
        train_path (str): Path to the processed training CSV file.

    Returns:
        tuple: (best_model_name, best_model_score, best_model_object)
    """
    try:
        logging.info("===== Model Training Started =====")

        # Step 1: Load dataset
        df = pd.read_csv(train_path)
        logging.info(f"Loaded training dataset: {df.shape}")

        # Label Encoding (store encoders)
        encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        # Define features (X) and target (y)
        X = df.drop("is_promoted", axis=1)    # All columns except target
        y = df["is_promoted"]                 # Target variable

        # Step 2: Train-Test Split
        # 80% training, 20% testing split
        # stratify=y (ensures the class distribution is same in train and test sets), random_state=42 (ensures reproducibility)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logging.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")

        # Step 3: Initialize Models
        # max_iter=500 (Maximum iterations to converge), random_state=42 (Reproducibility), eval_metric='logloss' (Loss function for classification)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
        }

        # Step 4: Cross-Validation on Initialize Models
        logging.info("Starting cross-validation for baseline models...")
        print("\n Cross-Validation Results:")
        cv_results = {}
        for name, model in models.items():
            X_train_cv = X_train.copy()

            # Scale features for Logistic Regression (sensitive to feature magnitude)
            if name == "Logistic Regression":
                scaler = StandardScaler()
                X_train_cv = scaler.fit_transform(X_train_cv)

            # cv=5: split data into 5 folds ( Perform 5-fold cross-validation), scoring='accuracy': evaluate based on accuracy
            scores = cross_val_score(model, X_train_cv, y_train, cv=5, scoring="accuracy")
            cv_results[name] = scores.mean()
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
            logging.info(f"{name} CV Accuracy: {scores.mean():.4f}")

        # Step 5: Hyperparameter Tuning
        logging.info("Starting hyperparameter tuning...")
        print("\n Hyperparameter Tuning...")

        # RandomForest hyperparameter grid ("n_estimators": No. of trees the model builds, "max_depth": Maximum depth of each tree)
        rf_params = {"n_estimators": [100, 200], "max_depth": [5, 10, None]}

        # GridSearchCV will test all combinations of these parameters 
        # (cv=5: Perform 3-fold cross-validation, scoring='accuracy': evaluate based on accuracy metric, n_jobs: Use all CPU cores)
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring="accuracy", n_jobs=-1)
        rf_grid.fit(X_train, y_train)

        best_rf = rf_grid.best_estimator_
        print(f"Best RF Params: {rf_grid.best_params_}")

        # XGBoost hyperparameter grid 
        # ("n_estimators": No. of sequential trees added,each correcting the errors of the prebious one, "learning_rate": controls how much each tree impacts the models prediction)
        xgb_params = {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}

        # GridSearchCV will test all combinations of these parameters
        xgb_grid = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), xgb_params, cv=3, scoring="accuracy", n_jobs=-1)
        xgb_grid.fit(X_train, y_train)

        best_xgb = xgb_grid.best_estimator_
        print(f"Best XGB Params: {xgb_grid.best_params_}")

        # Step 6: Final Model Evaluation
        final_models = {
            "RandomForest_Tuned": best_rf,
            "XGBoost_Tuned": best_xgb
        }

        os.makedirs("../models", exist_ok=True)
        best_model_name = None
        best_model_score = 0
        best_model = None

        logging.info("Final Test Set Evaluation...")
        print("\n Final Test Set Evaluation:")
        for name, model in final_models.items():
            # Train on full training data
            model.fit(X_train, y_train)

            # Predict on unseen test data
            y_pred = model.predict(X_test)

            # Evaluate performance
            acc = accuracy_score(y_test, y_pred) 
            
            print(f"\n{name} - Accuracy: {acc:.4f}")
            print("classification_report:\n", classification_report(y_test, y_pred))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

            logging.info(f"{name} Test Accuracy: {acc:.4f}")

            if acc > best_model_score:
                best_model_score = acc
                best_model_name = name
                best_model = model

        # Step 7: Save Best Model
        # Save the model to models folder for future predictions or deployment
        os.makedirs("../output/models", exist_ok=True)
        model_path = f"../output/models/{best_model_name}.pkl"
        joblib.dump(best_model, model_path)
        joblib.dump(encoders, "../output/models/encoders.pkl")

        logging.info(f"Best Model Saved: {model_path} with Accuracy {best_model_score:.4f}")
        print(f"\n Best Model Saved: {best_model_name} with Accuracy {best_model_score:.4f} at {model_path}")
        logging.info(f"Model training completed — Best Model: {best_model_name} with Accuracy {best_model_score:.4f}")
        print("===== Model Training Completed =====")

        return best_model_name, best_model_score, best_model

    except Exception as e:
        logging.error(f"Error in train_and_save_model: {e}")
        raise