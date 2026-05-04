# M7016H — Diabetes Prediction (Pima Indians)

Capstone project for the LTU course **M7016H: Artificial Intelligence within
the Healthcare System**. A binary classifier that predicts whether a patient
has diabetes from 8 routine clinical features, intended as a screening
decision-support tool.

## Dataset
Pima Indians Diabetes Dataset — 768 patients, 8 features, binary outcome
(0 = non-diabetic, 1 = diabetic).

## Pipeline
- 80/20 stratified train/test split
- Median imputation (zeros in 5 features encoded as missing)
- StandardScaler (fit on training data only)
- 5-fold cross-validated GridSearchCV optimizing ROC-AUC
- Compared models: Logistic Regression, Random Forest, XGBoost

## Results (test set, n=154)

| Model               | Sensitivity | Specificity | F1    | AUROC |
|---------------------|-------------|-------------|-------|-------|
| Logistic Regression | 0.704       | 0.730       | 0.639 | 0.811 |
| Random Forest       | 0.500       | 0.860       | 0.568 | 0.814 |
| XGBoost             | 0.556       | 0.860       | 0.612 | 0.821 |

Logistic Regression is preferred for the screening use case due to its
higher sensitivity, interpretability, and faster training time, despite
similar AUROC across all three models.

## Repo structure
- `M7016H_diabetes_project.ipynb` — main notebook
- `figures/` — generated plots
- `outputs/` — saved models and results CSVs

## Reproducibility
- Python 3.12, scikit-learn, XGBoost
- Random seed: 42
