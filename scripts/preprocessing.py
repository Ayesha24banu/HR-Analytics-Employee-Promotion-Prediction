# This script performs basic data cleaning and encoding
# It includes handling missing values and label encoding of categorical features
# Cleaned datasets are saved to cleaned_data/ directory and can be reused in notebooks or app.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import logging

# Function to preprocess both train and test data
# Business Question: Why do we preprocess?
# Explanation: Just like a resume is reformatted before sharing, raw data must be cleaned so that our ML model can understand it
# - Missing values are like blank fields in resumes — we fill them for clarity
# - Text data (e.g., 'Sales', 'HR') is converted to numbers so our model can 'read' it

def preprocess_and_save(train_path: str, test_path: str):
    """
    Cleans and encodes the train and test datasets and saves them to cleaned_data/ folder.

    Args:
        train_path (str): File path for the raw training CSV
        test_path (str): File path for the raw test CSV
    Returns:
        pd.DataFrame, pd.DataFrame: Cleaned train and test datasets.
    """

    try:
        logging.info("Starting preprocessing of HR Analytics data...")
        
        # Step 1: Load train and test CSV files
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info("Loaded raw training and test datasets.")
        logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # Step 2: Handle missing categorical values
        # Reason: Our model cannot handle blanks — we use 'Unknown' for categories and median for numerical
        train_df['education'] = train_df['education'].fillna('Unknown')
        test_df['education'] = test_df['education'].fillna('Unknown')

        # Handle missing numeric values using median
        median_rating = train_df['previous_year_rating'].median()
        train_df['previous_year_rating'] = train_df['previous_year_rating'].fillna(median_rating)
        test_df['previous_year_rating'] = test_df['previous_year_rating'].fillna(median_rating)
        logging.info("Filled missing values in education with 'Unknown' and previous_year_rating with median.")

        # Step 3: Drop employee_id for training data only (not useful for prediction)
        # Business Reason: This is just an identifier and has no relationship with promotion
        # Technical Reason: Including such unique IDs can lead to overfitting, where the model memorizes rather than learns patterns
        train_df.drop('employee_id', axis=1, inplace=True)
        logging.info("Dropped employee_id column.")

        # Step 4: Encode categorical columns using Label Encoding
        # Business Question: Why convert categories to numbers?
        # Explanation: Models understand numbers, not text — so we map 'HR', 'Sales' to 0, 1, etc.
        le = LabelEncoder()

        train_df['department'] = le.fit_transform(train_df['department'])
        test_df['department'] = le.transform(test_df['department'])

        train_df['region'] = le.fit_transform(train_df['region'])
        test_df['region'] = le.transform(test_df['region'])

        train_df['education'] = le.fit_transform(train_df['education'])
        test_df['education'] = le.transform(test_df['education'])

        train_df['gender'] = le.fit_transform(train_df['gender'])
        test_df['gender'] = le.transform(test_df['gender'])

        train_df['recruitment_channel'] = le.fit_transform(train_df['recruitment_channel'])
        test_df['recruitment_channel'] = le.transform(test_df['recruitment_channel'])
        logging.info("Applied Label Encoding to all categorical columns.")

        #Step 5: Save the cleaned datasets
        os.makedirs("../cleaned_data", exist_ok=True)
        train_df.to_csv("../cleaned_data/train_processed.csv", index=False)
        test_df.to_csv("../cleaned_data/test_processed.csv", index=False)
        logging.info("Saved cleaned datasets to cleaned_data/ folder.")

        return train_df, test_df
        
    except Exception as e:
        logging.error(f" Error in preprocess_and_save: {e}")
        raise