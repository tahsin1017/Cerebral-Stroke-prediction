# Comparative Deep Learning Framework Primitives and Ensemble Classifiers for Cerebrovascular Stroke Prediction under High Class Imbalance

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-emerald.svg)](https://www.python.org/)
[![Application](https://img.shields.io/badge/Domain-Clinical%20Informatics-darkred.svg)]()

## Abstract
Early prediction of ischemic and hemorrhagic stroke risk is vital for preventative clinical interventions, yet health informatics datasets are inherently constrained by extreme positive-class sparsity. This repository presents an experimental framework evaluating stroke risk profiles across 43,400 patient records. 

The project investigates two core areas of data science and neural engineering:
1. **Pipeline Integrity:** Establishing pre-processing pipelines where feature imputation, outlier treatment, and encoding layers are executed strictly within cross-validation boundaries to prevent data leakage and optimization bias.
2. **Deep Learning Framework Benchmarking:** A bottom-up pedagogical implementation of multi-layer neural network primitives from scratch using pure **NumPy mathematical arrays**, directly contrasted against production-ready abstractions in **PyTorch (nn.Module)** and **TensorFlow (Keras API)**.

---

## Dataset Profile & Pre-processing Pipelines
The empirical analysis utilizes a cohort of 43,400 clinical patient profiles containing demographic, historical, and physiological features. The target variable is heavily skewed, with true stroke events accounting for only **1.8%** of the dataset ($N_{\text{stroke}} = 626$ vs. $N_{\text{non-stroke}} = 34,094$).

### Missing Data & Imputation Mechanics
* `bmi`: 1,462 null observations
* `smoking_status`: 13,292 null observations

**Leakage Prevention:** Rather than applying global statistics across the entire dataset, missing continuous variables are handled via a train-validated `KNNImputer` ($k=5$). Missing categorical tags are inferred strictly using training fold mode replacement during pipeline execution.

### Pre-processing Engineering
* **Outlier Regularization:** Highly skewed physiological features (`bmi`, `avg_glucose_level`) are managed using a 1st/99th percentile Winsorization scheme computed from training distributions.
* **Vectorization:** Nominal inputs (`gender`, `work_type`, `Residence_type`, `smoking_status`) are converted into sparse numerical spaces using One-Hot Encoding, while ordinal relationships (`ever_married`) use direct mapping boundaries.
* **Imbalance Treatment:** Class distribution disparities are stabilized using SMOTE (Synthetic Minority Over-sampling Technique) applied strictly to the training fold to maintain baseline classification boundaries.

---

## Experimental Architecture & Frameworks
The baseline environment separates data using a stratified 80/20 train/test holdout distribution to preserve rare target fractions.
Raw Clinical Input ] ➔ [ Stratified Split ] ➔ [ Outlier Treatment & KNN Impute ]
│
┌──────────────────────────────────────────────┴──────────────────────────────────────────────┐
▼                                              ▼                                              ▼
[ Pure NumPy Engine ]                         [ TensorFlow Stack ]                            [ PyTorch Engine ]

Vectorized forward pass                     - Keras Sequential API                          - Custom nn.Module class

Manual Cross-Entropy Loss                   - Adam Optimizer, Binary Cross-Entropy          - Adam, BCELoss

Explicit matrix Backprop                    - Dense Stack (16 ➔ 8 ➔ 1)                      - Tensor autograd updates


### Regularization Profiling
To evaluate structural convergence under model complexity penalties, validation experiments track and isolate the effects of $L_1$ Lasso, $L_2$ Ridge, and Elastic Net regularizers, along with localized Early Stopping mechanisms to prevent validation loss divergence.

---

## Empirical Evaluation

### 1. Framework Convergence & Verification
The manual gradient derivations implemented via the NumPy matrix engine track uniformly alongside PyTorch and TensorFlow over identical scaling profiles, mathematically validating our custom backpropagation math:

| Deep Learning Engine | Hidden Layer Topology | Final Training Loss | Final Training MAE |
| :--- | :---: | :---: | :---: |
| **NumPy Matrix Primitives** | Baseline (16 ➔ 1) | 0.3024 | 0.0180 |
| **TensorFlow (Keras)** | Stacked (16 ➔ 8 ➔ 1) | 0.0713 | 0.0323 |
| **PyTorch (Core)** | Stacked (16 ➔ 8 ➔ 1) | 0.0793 | 0.0328 |

### 2. Clinical Classification Performance (Test Holdout)
Evaluating a highly imbalanced medical dataset purely on accuracy introduces severe optimization errors due to the 98.2% baseline majority. Because false negatives carry deep clinical risks, hyperparameter tuning sweeps—executed via Bayesian optimization using `Optuna`—explicitly targeted **Precision-Recall Area Under the Curve (PR-AUC)** and **Recall (Sensitivity)**.

* **Selected Optimal Model:** `RandomForestClassifier` (`max_depth=5`, `n_estimators=201`, `min_samples_split=8`, Stratified Class Weights).
* **Test Evaluation Profile:**
  * **ROC-AUC:** 0.8181
  * **PR-AUC (Average Precision):** 0.0895
  * **Recall (Sensitivity):** 29.94%

#### Test Set Confusion Matrix
                Predicted Negative    Predicted Positive
Actual Negative [         8,095          |       428        ]
Actual Positive [           110          |        47        ]


#### Feature Importances
Tree-based structural analysis shows that **Age**, **Average Glucose Level**, and **Hypertension** yield the highest statistical dependencies for identifying high-risk stroke events.

---

## Getting Started

### 1. Setup Source
```bash
git clone [https://github.com/tahsin1017/Cerebral-Stroke-prediction.git](https://github.com/tahsin1017/Cerebral-Stroke-prediction.git)
cd Cerebral-Stroke-prediction
2. Isolate Dependencies
Initialize a local virtual environment to decouple package installations:
python3 -m venv workspace_env
source workspace_env/bin/activate  # Windows: workspace_env\Scripts\activate

pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn tensorflow torch optuna notebook
3. Run Pipeline
Execute the master notebook workspace to evaluate data transformations, framework benchmarks, and comparative metrics:
jupyter notebook "ML Final Project.ipynb"
License
This work is released under the open-source MIT License — permitting redistribution, modification, and academic integration for extended healthcare informatics research.
