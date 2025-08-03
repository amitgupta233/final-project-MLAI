# Predicting Wholesale Loan Default using Hyperparameter Optimization

## Problem
This project aims to build a machine learning model to predict the probability of default in wholesale lending, a key component of credit risk management.

## Data
Source: [Link to dataset]
- Features: Borrower details, loan type, amount, duration, interest rate, etc.
- Target: Binary variable indicating default (1) or no default (0)

## Approach
- Model: XGBoostClassifier
- Optimization: Bayesian Optimization (Optuna)
- Evaluation: ROC-AUC, Accuracy, F1-score

## Results
- Best ROC-AUC: 0.86
- Final Parameters: [List top ones]

## Folder Structure
- `notebooks/`: Training and tuning notebooks
- `model_card.md`: Model details
- `data_sheet.md`: Dataset overview
