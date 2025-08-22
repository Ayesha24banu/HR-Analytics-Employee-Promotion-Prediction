"""
Prediction Script for HR Analytics
----------------------------------
Loads saved model, predicts promotions on the preprocessed test dataset,
and saves the predictions.
"""

import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def predict_and_save(model_path: str, test_path: str, output_path: str):
    """
    Load trained model, run predictions on already preprocessed test data,
    and save results with employee IDs.

    Args:
        model_path (str): Path to saved model (.pkl file)
        test_path (str): Path to preprocessed test CSV
        output_path (str): Path to save predictions CSV
    """
    try:
        logging.info("=== Prediction Started ===")

        # Step 1: Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        logging.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)   

        # Step 2: Load preprocessed test data
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data file not found at {test_path}")
        logging.info(f"Loading test dataset from {test_path}...")
        df_test = pd.read_csv(test_path)

        # Step 3: Encode bucket features 
        le = LabelEncoder()
        df_test['age_bucket'] = le.fit_transform(df_test['age_bucket'])
        df_test['tenure_bucket'] = le.fit_transform(df_test['tenure_bucket'])

        # Step 4: Prepare features (drop employee_id)
        employee_ids = df_test["employee_id"]
        X_test = df_test.drop(columns=["employee_id"], errors="ignore")

        # Step 5: Generate predictions
        logging.info("Generating predictions...")
        predictions = model.predict(X_test)

        # Step 6: Create results DataFrame
        results = pd.DataFrame({
            "employee_id": employee_ids,
            "is_promoted_predicted": predictions
        })

        # Preview first 10 predictions
        print("\n First 10 Predictions:")
        print(results.head(10))

        # Step 7: Save predictions
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_csv(output_path, index=False)

        logging.info(f"Predictions saved to {output_path}")
        print(f"\n Predictions saved to {output_path}")
        
        logging.info("=== Prediction Completed Successfully ===")

    except Exception as e:
        logging.error(f"Error in predict_and_save: {e}", exc_info=True)
        raise