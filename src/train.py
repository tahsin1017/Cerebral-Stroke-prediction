import pandas as pd
import numpy as np
import kagglehub
import os
import optuna
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("1. Downloading data...")
path = kagglehub.dataset_download("shashwatwork/cerebral-stroke-predictionimbalaced-dataset")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

# Drop 'id' as it has no predictive power
X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

print("2. Splitting data (No leakage)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("3. Building Pipeline...")
num_cols = ['age', 'avg_glucose_level', 'bmi']
cat_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

num_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

print("4. Starting Optuna tuning (with 3-fold CV)...")
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 150)
    max_depth = trial.suggest_int('max_depth', 5, 15)
    
    # This Pipeline guarantees no leakage! SMOTE is applied ONLY to training folds.
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1))
    ])
    
    # Maximize PR-AUC (average_precision) because of severe class imbalance
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='average_precision')
    return cv_scores.mean()

# Run Optuna (10 trials to keep it fast for today)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print(f"\nBest Optuna Params: {study.best_params}")

print("\n5. Training final model with best params...")
best_params = study.best_params
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(**best_params, random_state=42, n_jobs=-1))
])
final_model.fit(X_train, y_train)

print("\n6. Evaluating on TEST set...")
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Test PR-AUC: {average_precision_score(y_test, y_pred_proba):.4f}")
print("Done!")
