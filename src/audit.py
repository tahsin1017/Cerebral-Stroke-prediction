import pandas as pd
import numpy as np
import kagglehub
import os
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, average_precision_score

print("1. Loading data...")
path = kagglehub.dataset_download("shashwatwork/cerebral-stroke-predictionimbalaced-dataset")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

X = df.drop(columns=['id', 'stroke'])
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("2. Building Pipeline and Training Model...")
num_cols = ['age', 'avg_glucose_level', 'bmi']
cat_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())]), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

# Using the best params found in Day 3
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=136, max_depth=5, random_state=42, n_jobs=-1))
])

model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("3. Generating SHAP Plot...")
# We must transform X_test for SHAP to understand it
X_test_transformed = model.named_steps['preprocessor'].transform(X_test)
# Get feature names after OneHotEncoding
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

explainer = shap.TreeExplainer(model.named_steps['classifier'])
shap_values = explainer.shap_values(X_test_df)

# Handle both old (list) and new (3D array) shap versions
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values[:, :, 1]

plt.figure()
shap.summary_plot(sv, X_test_df, show=False)
plt.tight_layout()
plt.savefig('figures/shap_summary.png', dpi=300, bbox_inches='tight')
print("SHAP plot saved to figures/shap_summary.png!")

print("4. Running Fairness Audit...")
# Create age groups
X_test_copy = X_test.copy()
X_test_copy['y_true'] = y_test
X_test_copy['y_proba'] = y_pred_proba
X_test_copy['y_pred'] = (y_pred_proba >= 0.5).astype(int)
X_test_copy['age_group'] = pd.cut(X_test_copy['age'], bins=[0, 40, 60, 120], labels=['<40', '40-60', '>60'])

# Calculate metrics by Gender
gender_audit = X_test_copy.groupby('gender').apply(
    lambda x: pd.Series({
        'Recall': recall_score(x['y_true'], x['y_pred'], zero_division=0),
        'PR-AUC': average_precision_score(x['y_true'], x['y_proba'])
    })
)
print("\n--- Fairness Audit: Gender ---")
print(gender_audit)

# Calculate metrics by Age Group
age_audit = X_test_copy.groupby('age_group').apply(
    lambda x: pd.Series({
        'Recall': recall_score(x['y_true'], x['y_pred'], zero_division=0),
        'PR-AUC': average_precision_score(x['y_true'], x['y_proba'])
    })
)
print("\n--- Fairness Audit: Age Group ---")
print(age_audit)
