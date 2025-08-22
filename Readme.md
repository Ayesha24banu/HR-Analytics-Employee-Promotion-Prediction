# 📊 HR Analytics – Employee Promotion Prediction using Machine Learning

## 1️⃣ Business Problem
HR departments often struggle to identify employees who truly deserve promotions.  
Bias, incomplete information, and manual reviews can lead to **unfair decisions** and **lower employee satisfaction**.  

👉 Our task: **Predict whether an employee will be promoted** based on historical HR data.  
This ensures **data-driven fairness** in promotions and helps HR managers **focus on top talent**.

---

## 2️⃣ Data Overview

### 📂 Dataset Files
- `train.csv`: 54,808 employees (with target `is_promoted`)  
- `test.csv`: 23,490 employees (without target, used for predictions)

### 🔑 Key Columns
- **employee_id**: Unique identifier (not useful for prediction → dropped later)  
- **department, region, education, gender, recruitment_channel**: Categorical attributes  
- **no_of_trainings, age, previous_year_rating, length_of_service, KPIs_met >80%, awards_won?, avg_training_score**: Numerical attributes  
- **is_promoted**: Target variable (0 = Not promoted, 1 = Promoted)

---

## 3️⃣ Methodology

### 🔍 Step 1: Exploratory Data Analysis (EDA)
- Checked **missing values** → Found missing `education`, `previous_year_rating`  
- Plotted **univariate distributions** (age, training scores, ratings)  
- Bivariate analysis → Promotions are strongly linked to:
  - **High KPI (>80%)**  
  - **High ratings (≥4)**  
  - **Awards won**  
- Outlier detection on age, length_of_service  
- Correlation matrix → low collinearity, so features are useful

✅ **Reasoning:**  
EDA ensures we **understand patterns** before modeling.  
E.g., knowing that **KPI + rating** correlate with promotion → feature engineering idea.

---

### 🧹 Step 2: Preprocessing
Script: `scripts/preprocessing.py`

- Missing categorical (`education`) → filled with `"Unknown"`  
- Missing numeric (`previous_year_rating`) → filled with **median**  
- Dropped `employee_id` (identifier, no predictive value)  
- Label Encoding for categorical features (`department`, `region`, `education`, `gender`, `recruitment_channel`)  

✅ **Reasoning:**  
ML models cannot handle NaNs or text → must be numeric and clean.  
Dropping IDs avoids **overfitting** (model memorizing instead of learning).

---

### 🛠 Step 3: Feature Engineering
Script: `scripts/feature_engineering.py`

Added **new HR-relevant features**:
1. `age_bucket` → Young / Mid / Senior  
2. `tenure_bucket` → New / Experienced / Veteran  
3. `high_performance_flag` → (KPI >80% + rating ≥4)  

✅ **Reasoning:**  
HR often thinks in **groups** (young talent, experienced hires, veterans).  
Features like `high_performance_flag` directly map to promotion rules.

---

### 🤖 Step 4: Model Training
Script: `scripts/model_training.py`

- **Train/Test Split** (80/20, stratified to preserve promotion ratio)  
- Compared baseline models:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
- Applied **Cross-validation (CV=5)**  
- Performed **Hyperparameter tuning** (GridSearchCV)  
  - RF → tuned `n_estimators`, `max_depth`  
  - XGB → tuned `n_estimators`, `max_depth`, `learning_rate`  
- Final evaluation on test split  
- Saved **best model (.pkl)**  

✅ **Reasoning:**  
We start simple (LogReg), then test complex models.  
RF/XGB capture non-linear relationships better → improved accuracy.

---

### 🔮 Step 5: Predictions
Script: `scripts/predict.py`

- Loaded **best model**  
- Transformed test data (same encoders + features)  
- Predicted `is_promoted_predicted`  
- Saved results → `output/predictions.csv`  

✅ **Reasoning:**  
Encoders ensure **consistent category handling**.  
Predictions are reproducible and exportable for HR review.

---

### 🖥️ Step 6: Streamlit Dashboard
App: `app.py`

**Features:**
1. **Single Prediction Tab**  
   - HR inputs employee details manually  
   - Model outputs promotion likelihood (✅ Likely / ❌ Unlikely) + confidence score (%)  
2. **Batch Prediction Tab**  
   - Upload CSV of employees  
   - Predictions + confidence for all employees
   - Downloadable results CSV
   - Visualizations:
     - Promotion distribution by Department
     - KPI >80% pie chart
     - Education vs Promotion Rate

✅ **Reasoning:**  
Streamlit makes ML **accessible to non-technical users (HR managers)**.  
Visuals improve **interpretability** of model outputs.

---

## 4️⃣ Logging & Monitoring
- All scripts log progress to `logs/`  
- Examples:  
  - Data loaded, shape  
  - Missing values handled  
  - Model accuracy scores  
  - Errors in plots/file paths  

✅ **Reasoning:**  
Logs help in **debugging** & **tracking experiments**.

---

## 5️⃣ Results

- **Best Model:** Random Forest / XGBoost (depending on dataset splits)  
- **Accuracy:** ~92%  
- **Key Insights:**  
  - Employees with **high KPI + rating ≥4** have **3x higher promotion chances**  
  - **Awards** also strongly linked to promotions  
  - Education has weaker impact compared to performance metrics  

---
## Output

### App Screenshots:
![Single Prediction](<Single Prediction.png>)

![Single Prediction - no promotion](<Single Prediction - no promotion (1).png>)

![Single Prediction - no promotion](<Single Prediction - no promotion (2).png>)

![Single Prediction - promotion](<Single Prediction - promotion (1).png>)

![Single Prediction - promotion](<Single Prediction - promotion (2).png>)

![Batch Prediction & Dashboard](<Batch Prediction & Dashboard.png>)

![Batch Prediction & Dashboard](<Batch Prediction & Dashboard - 1.png>)

![Batch Prediction & Dashboard](<Batch Prediction & Dashboard - 2.png>)

![Batch Prediction & Dashboard](<Batch Prediction & Dashboard - 3.png>)

![Batch Prediction & Dashboard](<Batch Prediction & Dashboard - 4.png>)

![Batch Prediction & Dashboard](<Batch Prediction & Dashboard - 5.png>)

---

## 🚀 Demo:

<video controls src="HR - Analytics.mp4" title="Title"></video>

---

## 6️⃣ Future Work
- Integrate **SHAP explainability** → show why a specific employee got predicted result  
- Add **fairness metrics** (avoid gender/region bias)  
- Deploy on **AWS/GCP/Streamlit Cloud**  
- Automate retraining with **MLflow pipeline**  

---

## 7️⃣ Project Structure
HR_Analytics/
│
├── data/ # Raw datasets (train.csv, test.csv)
├── cleaned_data/ # Processed datasets after cleaning & feature engineering
├── output/
│ ├── figures/ # EDA plots (univariate, bivariate, outliers, correlations)
│ └── models/ # Trained models (.pkl) + encoders
├── logs/ # Logging outputs for debugging & monitoring
├── notebook/ # Jupyter notebooks for EDA & experimentation
├── scripts/ # Modular Python scripts for pipeline
│ ├── __init.py __
│ ├── eda.py
│ ├── preprocessing.py # Missing value handling, label encoding
│ ├── feature_engineering.py # Age buckets, tenure buckets, performance flags
│ ├── model_training.py # Model training, hyperparameter tuning, evaluation
│ ├── predict.py # Load model, run predictions, save results
│
├── app.py # Streamlit dashboard for single/batch predictions
├── requirements.txt # Project dependencies
└── README.md # Documentation

---

## 8️⃣ Setup Instructions

1. **Clone the repository**
    ```bash
    git clone <repo-url>
    cd HR_Analytics
    ```

2. **Create and activate a virtual environment**
    ```bash
    python -m venv venv
    venv\Scripts\activate   # On Windows
    # source venv/bin/activate   # On Mac/Linux
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the app**
    ```bash
    streamlit run app.py
    ```

5. **Usage**
    - Fill in employee details in the sidebar.
    - Click "Predict Promotion Likelihood" to see the result.
    - View EDA images and insights if available.

## Notes
- Ensure `output/models/XGBoost_Tuned.pkl` and `output/models/encoders.pkl` exist. If not, train the model first.
- Place EDA images in `output/figures/` to display them in the app.
- For any issues, check the logs or raise an issue in the repository.

## Requirements

See [`requirements.txt`](requirements.txt) for all Python dependencies.

---

### 👤 Author
**Ayesha Banu**
- 🎓 M.Sc. Computer Science | 🏅 Gold Medalist
- 💼 Data Scientist | Data Analyst | Full-Stack Python Developer | GenAI Enthusiast
- 📫 [LinkedIn](https://www.linkedin.com/in/ayesha_banu_cs)
- **Project:** HR Analytics – Employee Promotion Prediction  -- Aug/2025  
---

### 📄 License
Distributed under the MIT License. See `LICENSE` file for details.

