# credit_default_xgboost_tuning.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import warnings

warnings.filterwarnings("ignore")

# Step 1: Download dataset
path = kagglehub.dataset_download("wordsforthewise/lending-club")
print("Dataset downloaded to:", path)

# Step 2: Load data
df = pd.read_csv(os.path.join(path, "accepted_2007_to_2018Q4.csv"), low_memory=False)

# Step 3: Preprocessing
df = df[['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
         'delinq_2yrs', 'revol_util', 'total_acc', 'loan_status']]
df.dropna(inplace=True)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Bayesian Hyperparameter Search
search_space = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0)
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

opt = BayesSearchCV(
    estimator=xgb,
    search_spaces=search_space,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    random_state=42,
    verbose=0
)

opt.fit(X_train, y_train)
print("Best Parameters:", opt.best_params_)

# Step 5: Evaluation
y_pred = opt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
