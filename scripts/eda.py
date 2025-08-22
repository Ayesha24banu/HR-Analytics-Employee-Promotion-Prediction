"""
Exploratory Data Analysis (EDA) for HR Analytics Project
--------------------------------------------------------
This script helps us understand the data before building models.

We use the RAW dataset (train.csv) so categories are easy to interpret.

Steps:
1. Check missing values (data quality check)
2. Look at single columns (Univariate Analysis)
3. Compare columns with the target (Bivariate Analysis)
4. Check relationships between numbers (Multivariate Analysis)
5. Look for unusual values (Outlier Detection)
"""

import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# Helper to sanitize filenames
def safe_filename(name: str) -> str:
    """Replace invalid filename characters with underscores."""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '%', ' ']
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name

# Helper to annotate counts and percentages
def annotate_bars(ax, total):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}\n({height/total:.1%})',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, color='black')

# Helper: annotate counts only
def annotate_bars_counts(ax):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, color='black')


# Step 1: Missing Values
def check_missing_values(data):
    try:
        logging.info("Checking missing values...")
        print("\n[1] Missing Values Check")
        missing = data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if not missing.empty:
            print(missing.to_frame(name="Missing Count").assign(Percent=lambda x: x['Missing Count'] / len(data) * 100))
            plt.figure(figsize=(8, 5))
            missing.plot(kind='bar', color='skyblue')
            plt.title("Missing Values by Column", fontsize=14)
            plt.ylabel("Number of Missing Values", fontsize=12)
            plt.tight_layout()
            os.makedirs("../output/figures", exist_ok=True)
            plt.savefig("../output/figures/missing_values.png")
            plt.show()
            plt.close()
        else:
            logging.info("No missing values found.")
            print(" No missing values found.")
    except Exception as e:
        logging.error(f"Error in check_missing_values: {e}")
        print(f"⚠ Error in check_missing_values: {e}")


# Step 2: Univariate Analysis
def univariate_analysis(data):
    try:
        logging.info("Doing univariate analysis...")
        print("\n[2] Univariate Analysis")

        # Categorical columns
        categorical = data.select_dtypes(include='object').columns
        for col in categorical:
            try:
                plt.figure(figsize=(8, 5))
                counts = data[col].value_counts()
                ax = sns.countplot(x=col, data=data, order=counts.index, palette="Blues")
                if col == 'region':
                    annotate_bars_counts(ax)       # only counts for region
                else:
                    annotate_bars(ax, len(data))   # counts + percentage for others

                plt.title(f"{col.title()} Distribution", fontsize=14)
                plt.ylabel("Number of Employees", fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"../output/figures/univariate_{safe_filename(col)}.png")
                plt.show()
                plt.close()
            except Exception as e:
                logging.error(f"Error plotting univariate for {col}: {e}")

        # Numerical columns
        numerical = data.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical:
            if col not in ['employee_id', 'is_promoted']:
                try:
                    plt.figure(figsize=(8, 5))
                    sns.histplot(data[col], kde=True, color="skyblue")
                    plt.axvline(data[col].mean(), color='red', linestyle='--', label=f"Mean: {data[col].mean():.2f}")
                    plt.axvline(data[col].median(), color='green', linestyle='-', label=f"Median: {data[col].median():.2f}")
                    plt.title(f"Distribution of {col.title()}", fontsize=14)
                    plt.ylabel("Count", fontsize=12)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"../output/figures/univariate_{safe_filename(col)}.png")
                    plt.show()
                    plt.close()
                except Exception as e:
                    logging.error(f"Error plotting univariate for {col}: {e}")
    except Exception as e:
        logging.error(f"Error in univariate_analysis: {e}")
        print(f"⚠ Error in univariate_analysis: {e}")


# Step 3: Bivariate Analysis
def bivariate_analysis(data):
    try:
        logging.info("Doing bivariate analysis...")
        print("\n[3] Bivariate Analysis")

        if 'is_promoted' not in data.columns:
            logging.warning("No target column found.")
            print("⚠ No target column 'is_promoted' found.")
            return

        # Categorical vs Target
        categorical = data.select_dtypes(include='object').columns
        for col in categorical:
            try:
                plt.figure(figsize=(8, 5))
                ax = sns.countplot(x=col, hue='is_promoted', data=data, palette={0: "red", 1: "green"})
                if col == 'region':
                    annotate_bars_counts(ax)     # only counts for region
                else:
                    annotate_bars(ax, len(data)) # counts + percentage for others

                plt.title(f"Promotion Rate by {col.title()}", fontsize=14)
                plt.ylabel("Number of Employees", fontsize=12)
                plt.xticks(rotation=45)
                plt.legend(title="Promoted", labels=["No", "Yes"])
                plt.tight_layout()
                plt.savefig(f"../output/figures/bivariate_{safe_filename(col)}.png")
                plt.show()
                plt.close()
            except Exception as e:
                logging.error(f"Error plotting bivariate for {col}: {e}")

        # Numerical vs Target
        numerical = data.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical:
            if col != 'employee_id':
                try:
                    plt.figure(figsize=(8, 5))
                    sns.boxplot(x='is_promoted', y=col, data=data, palette=["red", "green"], showmeans=True,
                                meanprops={"marker": "o", "markerfacecolor": "blue", "markeredgecolor": "black", "markersize": 7})
                    plt.title(f"{col.title()} by Promotion Status", fontsize=14)
                    plt.xlabel("Promotion Status", fontsize=12)
                    plt.ylabel(col.title(), fontsize=12)
                    plt.xticks([0, 1], ["No", "Yes"])
                    # Add grid for easy reading
                    plt.grid(axis='y', linestyle='--', alpha=0.6)

                    # Add annotation for medians
                    medians = data.groupby('is_promoted')[col].median()
                    for i, median in enumerate(medians):
                        plt.text(i, median, f'Median: {median:.1f}', ha='center', va='bottom', color='black', fontsize=9)
                    plt.tight_layout()
                    plt.savefig(f"../output/figures/bivariate_{safe_filename(col)}.png")
                    plt.show()
                    plt.close()
                except Exception as e:
                    logging.error(f"Error plotting numerical bivariate for {col}: {e}")
    except Exception as e:
        logging.error(f"Error in bivariate_analysis: {e}")
        print(f"⚠ Error in bivariate_analysis: {e}")

# Step 4: Multivariate Analysis
def multivariate_analysis(data):
    try:
        logging.info("Doing multivariate analysis...")
        print("\n[4] Multivariate Analysis - Correlation Heatmap")
        numerical = data.select_dtypes(include=['int64', 'float64']).drop(columns=['employee_id'], errors='ignore')
        plt.figure(figsize=(10, 8))
        sns.heatmap(numerical.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
        plt.title("Correlation Heatmap", fontsize=14)
        plt.tight_layout()
        plt.savefig("../output/figures/correlation_heatmap.png")
        plt.show()
        plt.close()
    except Exception as e:
        logging.error(f"Error in multivariate_analysis: {e}")
        print(f"⚠ Error in multivariate_analysis: {e}")

# Step 5: Outlier Detection
def detect_outliers(data):
    try:
        logging.info("Checking for outliers...")
        print("\n[5] Outlier Detection")
        numerical = data.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical:
            if col != 'employee_id':
                try:
                    plt.figure(figsize=(8, 5))
                    sns.boxplot(x=data[col], color="skyblue", flierprops=dict(markerfacecolor='orange', marker='o'))
                    plt.title(f"Outlier Check - {col.title()}", fontsize=14)
                    plt.xlabel(col.title(), fontsize=12)
                    plt.tight_layout()
                    plt.savefig(f"../output/figures/outliers_{safe_filename(col)}.png")
                    plt.show()
                    plt.close()
                except Exception as e:
                    logging.error(f"Error plotting outliers for {col}: {e}")
    except Exception as e:
        logging.error(f"Error in detect_outliers: {e}")
        print(f"⚠ Error in detect_outliers: {e}")