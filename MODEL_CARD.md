# Model Card — Cerebral Stroke Prediction

## Intended Use
This model is a research prototype for stroke risk stratification. It is **not** intended for clinical deployment or medical diagnosis. It is designed to demonstrate rigorous machine learning practices on highly imbalanced tabular data.

## Dataset
- **Source:** Kaggle (fedesoriano Stroke Prediction Dataset)
- **N = 5,110** (Note: Update this if your downloaded dataset has 43,400 rows)
- **Positive rate:** ~1.8% (Severe class imbalance)
- **Features:** Age, hypertension, heart disease, avg_glucose_level, bmi, smoking_status, gender, work_type, residence_type, ever_married.

## Model Architecture
- **Algorithm:** RandomForestClassifier
- **Hyperparameter Tuning:** Optuna (10 trials, maximizing PR-AUC via 3-fold Cross-Validation)
- **Preprocessing Pipeline:** 
  - `KNNImputer` (k=5) for missing `bmi`
  - `StandardScaler` for numerical features
  - `OneHotEncoder` for categorical features
- **Class Imbalance Handling:** `SMOTE` applied strictly to training folds within the Cross-Validation pipeline to prevent data leakage.

## Evaluation Metrics (Held-out Test Set)
- **ROC-AUC:** 0.8156
- **PR-AUC:** 0.0611
*(Note: PR-AUC is a more informative metric than ROC-AUC for highly imbalanced datasets.)*

## Limitations
- **Retrospective Data:** The model is trained on historical data and cannot infer causality.
- **Demographic Bias:** The dataset may underrepresent certain populations. Performance across demographic subgroups (e.g., gender, age) should be audited before any real-world application.
- **No External Validation:** The model has not been tested on an independent cohort from a different hospital or country.

## Ethical Considerations
Because this model predicts a life-threatening condition, any downstream application must prioritize high recall (sensitivity) over precision. False negatives are far more dangerous than false positives in this context.
