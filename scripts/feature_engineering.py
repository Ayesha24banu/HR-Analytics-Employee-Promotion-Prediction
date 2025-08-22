"""
Business Context:
-----------------
Why create new features?
Just like HR might create a "high potential" tag by combining multiple performance indicators,
we can generate new variables from the raw data that may capture hidden patterns.
"""

import pandas as pd
import logging

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds additional features to the HR dataset for better prediction and explainability.

    Args:
        df (pd.DataFrame): The original DataFrame (can be train or test).

    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    df = df.copy()
    logging.info("Starting feature engineering...")

    # 1. Age Buckets
    # Business Reason: Grouping age can make trends easier to spot in reports and discussions.
    # For example: Are younger employees less likely to be promoted compared to senior ones?
    if 'age' in df.columns:
        df['age_bucket'] = pd.cut(
            df['age'],
            bins=[17, 25, 35, 60],  # realistic HR ranges
            labels=['Young', 'Mid', 'Senior']
        )
        logging.info("Added 'age_bucket' feature.")

    # 2. Tenure Buckets
    # Business Reason: Time in company can influence promotion likelihood.
    # HR might notice that 'Experienced' employees get promoted more often than 'New' hires.
    if 'length_of_service' in df.columns:
        df['tenure_bucket'] = pd.cut(
            df['length_of_service'],
            bins=[-1, 2, 5, 40],
            labels=['New', 'Experienced', 'Veteran']
        )
        logging.info("Added 'tenure_bucket' feature.")

    # 3. High Performance Flag
    # Business Reason: The problem statement highlights KPI completion (>80%) and past performance as major promotion criteria. 
    # Combining them into one flag makes it easy for the model to recognize top performers early.
    if 'KPIs_met >80%' in df.columns and 'previous_year_rating' in df.columns:
        df['high_performance_flag'] = ((df['KPIs_met >80%'] == 1) & (df['previous_year_rating'] >= 4)).astype(int)
        logging.info("Added 'high_performance_flag' feature.")

    logging.info("Feature engineering complete.")
    return df

def feature_engineer_and_save(train_path: str, test_path: str):
    """
    Reads preprocessed train and test datasets, applies feature engineering,
    and overwrites them in cleaned_data/ folder.

    Args:
        train_path (str): Path to preprocessed train CSV
        test_path (str): Path to preprocessed test CSV
    """
    try:
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info(f"Initial train shape: {train_df.shape}, test shape: {test_df.shape}")

        # Apply feature engineering
        train_df = add_features(train_df)
        test_df = add_features(test_df)

        # Save updated files (overwrite processed files at given paths)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info(f"Feature engineered train saved to: {train_path}")
        logging.info(f"Feature engineered test saved to: {test_path}")

        return train_df, test_df

    except Exception as e:
        logging.error(f"Error in feature_engineer_and_save: {e}")
        raise