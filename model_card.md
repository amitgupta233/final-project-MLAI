# Model Card: XGBoost for Loan Default Prediction

## Model Type
- XGBoost Classifier

## Objective
To predict the probability of default in wholesale loan customers.

## Input Features
- Loan amount, interest rate, borrower industry, loan duration, etc.

## Output
- Binary: 1 = default, 0 = no default

## Training Process
- Bayesian Optimization using Optuna
- 5-fold cross-validation on training set
