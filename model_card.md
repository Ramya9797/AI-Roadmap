# Customer Churn Prediction Model

## 1. Overview

This model predicts whether a customer is likely to churn.

## 2. Business Objective

The objective is to identify customers who have a high probability
of leaving the service so that retention teams can take preventive
actions.

## 3. Input Features

The model uses:

- tenure
- MonthlyCharges
- TotalCharges
- Contract
- PaymentMethod
- InternetService

## 4. Target

Target:

- 0 = Customer does not churn
- 1 = Customer churns

## 5. Preprocessing

Numerical features:

- StandardScaler

Categorical features:

- OneHotEncoder
- handle_unknown="ignore"

## 6. Model

Algorithm:

Logistic Regression

The preprocessing and model are stored together using an
scikit-learn Pipeline.

## 7. Evaluation

Metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

See outputs/metrics.json for the actual results.

## 8. Error Analysis

Misclassified records are saved in:

outputs/misclassified.csv

The error analysis helps identify customer patterns where the
model makes incorrect predictions.

## 9. API

The model is exposed using FastAPI.

Endpoints:

GET /health

POST /predict

## 10. Limitations

The model depends on the quality and representativeness of the
training data.

Customer behavior may change over time, causing model drift.

The model should therefore be monitored after deployment.

## 11. Production Considerations

Potential improvements include:

- MLflow experiment tracking
- Model registry
- Docker
- Kubernetes
- Prometheus monitoring
- Data drift monitoring
- Automated retraining
- CI/CD