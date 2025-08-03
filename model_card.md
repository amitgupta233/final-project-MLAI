# 📊 Model Card: Credit Default Prediction using XGBoost

## 📝 Model Details
- **Model Type**: XGBoost Classifier (Gradient Boosted Trees)
- **Use Case**: Predicting likelihood of loan default based on Lending Club data
- **Problem Type**: Binary Classification (`Default` vs. `No Default`)
- **Optimization Technique**: Bayesian Optimization (for hyperparameter tuning)

---

## 📚 Intended Use
This model is intended to support wholesale lending credit risk analysis by predicting the likelihood of a borrower defaulting on a loan. It is useful for credit scoring, risk profiling, and portfolio risk management.

---

## 📈 Performance Metrics (Validation Set)

| Metric      | Value     |
|-------------|-----------|
| Accuracy    | 71.96%    |
| Precision   | 72.72%    |
| Recall      | 73.33%    |
| F1 Score    | 73.02%    |
| AUC (Train) | ~0.79     |

- These metrics are based on a validation set split from the Lending Club dataset.
- Evaluation was performed using the confusion matrix and ROC AUC.

---

## 📊 Confusion Matrix

|                         | Predicted No Default | Predicted Default |
|-------------------------|----------------------|-------------------|
| **Actual No Default**   | 143,702              | 60,169            |
| **Actual Default**      | 58,371               | 160,482           |

---

## ⚙️ Hyperparameters Tuned

The following hyperparameters were optimized using Bayesian Optimization:

- `max_depth`
- `learning_rate`
- `n_estimators`
- `gamma`
- `min_child_weight`
- `subsample`
- `colsample_bytree`

---

## 🧪 Training and Validation

- **Train/Test Split**: 80/20 stratified split
- **Cross-validation**: Stratified 5-fold CV during hyperparameter tuning
- **Data Source**: [Kaggle Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

---

## 🔍 Observations and Recommendations

- The model performs reasonably well with balanced precision and recall.
- There is room for improvement via:
  - Feature engineering
  - Handling class imbalance more explicitly (SMOTE or class weighting)
  - Comparing with other models (e.g., LightGBM, Logistic Regression)
- Business decisions could consider thresholds beyond 0.5 for adjusting sensitivity.

---

## 📦 Limitations

- **Bias in Lending Club data**: It reflects specific historical decisions by underwriters.
- **Missing external factors**: Macroeconomic indicators, borrower history outside platform.
- **Not intended for live deployment without further validation and monitoring.**

---

## 👤 Authors

- Amit Gupta  
- Capstone Project: Machine Learning for Credit Risk  
- August 2025

