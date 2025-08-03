# Predicting Wholesale Loan Default using Hyperparameter Optimization

## 📝 Project Summary (Non-Technical)

This project builds a machine learning model to predict whether a borrower will default on a loan, using historical Lending Club data. The model uses XGBoost and is optimized using Bayesian hyperparameter tuning. The results show strong performance with an AUC score above 0.84, enabling better identification of risky loan applicants. This can help financial institutions make informed lending decisions, reduce default rates, and improve profitability. The notebook presents the process step-by-step and includes insights from the confusion matrix and feature importance to explain how the model makes decisions.

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
