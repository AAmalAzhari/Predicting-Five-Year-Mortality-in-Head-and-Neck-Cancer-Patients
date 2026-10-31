# Predicting Five-Year Mortality in Head and Neck Cancer Patients: A Machine Learning Pipeline in R using `caret`

An end-to-end binary classification project using `caret` and `recipes` on a **synthetic multimodal cohort**
(1,200 patients, 45 predictors). Ten algorithms are tuned with cross-validation, compared on AUC, and the
selected model is evaluated once on a held-out test set.

[View the rendered report](https://aamalazhari.github.io/Predicting-Five-Year-Mortality-in-Head-and-Neck-Cancer-Patients/)

## Project Overview

| Step | What is done |
|:--|:--|
| 1. Data quality | Missing-value audit, detection of physiologically impossible values (age, BMI, pack-years) |
| 2. Split | Stratified 80/20 train/test split; the test set is untouched until the end |
| 3. EDA | Distributions, outcome versus predictors, skewness and correlation (training set only) |
| 4. Pre-processing | Leakage-free `recipes` pipeline: impossible values → NA, KNN imputation, dummy coding, Yeo-Johnson, scaling, correlation filter |
| 5. Modelling | Ten algorithms tuned by 5-fold cross-validation on shared folds |
| 6. Selection | Final model chosen by cross-validated AUC|
| 7. Evaluation | ROC, confusion matrix, calibration, variable importance, decision-threshold analysis |

**Models:** elastic net (`glmnet`), partial least squares (`pls`), regularised discriminant analysis, naive Bayes,
k-nearest neighbours (`knn`), shrunken centroids, gradient boosting (`gbm`), XGBoost (custom `caret` model),
neural networks (`nnet`, `mlp`).

## Key Results for the Final Model


| Item | Value |
|:--|:--|
| Final model | `pls` |
| Cross-validated AUC (training set) | 0.842 |
| Test AUC | 0.78|
| Sensitivity / specificity at threshold 0.5 (test) | 0.60 / 0.85 |


## Skills Demonstrated

- Reproducible reporting with Quarto: narrative text whose numbers update from the model output
- Leakage-free pre-processing and cross-validation with `recipes` and `caret`
- Writing a custom `caret` model (XGBoost) to work around a package incompatibility
- Model comparison, calibration, over-fitting checks and threshold selection
- Handling missing data, data-entry errors and correlated predictors

## Repository structure

```text
├── analysis/Mortality_Prediction_ML.qmd      # full report (code + narrative)
├── scripts/Mortality_Prediction_ML.R         # code only, extracted from the report
├── data/synthetic_hnscc_data.csv             # synthetic cohort (safe to share)
├── docs/index.html                           # rendered report (GitHub Pages)
└──outputs/                                  # result graphs (PNG)
```

## Data

The data is synthetic, mimicing the problems of clinical data (missing values, impossible data-entry errors, skewed and correlated variables). 


## How to Reproduce

1. Clone the repository and open `Predicting-Five-Year-Mortality-in-Head-and-Neck-Cancer-Patients` in RStudio.
2. Restore the packages: `renv::restore()`
3. Render the report: `quarto render analysis/Mortality_Prediction_ML.qmd`

The first render tunes ten models with 5-fold cross-validation and takes roughly 10–30 minutes. The fitted
models are cached in `cache/models.rds` (not stored in this repository), so later renders take seconds.
Delete that file to force a re-fit.

**Requirements:** R ≥ 4.3, Quarto ≥ 1.4.

## Author

**Abdulrahman Alazhari**: [LinkedIn](https://www.linkedin.com/in/aaazhari/) · [Email](mailto:aamalazhari@hotmail.com)
