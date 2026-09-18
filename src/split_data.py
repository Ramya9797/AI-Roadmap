# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score


# # 1. Create dataset
# data = {
#     "age": [25, 35, 45, 52, 23, 40, 60, 30, 48, 55],
#     "monthly_bill": [50, 70, 90, 120, 45, 80, 150, 60, 110, 130],
#     "support_calls": [1, 2, 4, 6, 0, 3, 8, 1, 5, 7],
#     "tenure": [24, 36, 12, 5, 30, 20, 3, 40, 8, 4],
#     "churn": [0, 0, 1, 1, 0, 0, 1, 0, 1, 1]
# }

# df = pd.DataFrame(data)

# print("\nDataset:")
# print(df)


# # 2. Separate features and target
# X = df.drop("churn", axis=1)
# y = df["churn"]


# # 3. Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# print("\nTraining records:", len(X_train))
# print("Testing records:", len(X_test))


# # 4. Create ML pipeline
# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", LogisticRegression())
# ])


# # 5. Train model
# pipeline.fit(X_train, y_train)

# print("\nModel training completed.")


# # 6. Make predictions
# predictions = pipeline.predict(X_test)

# print("\nActual values:")
# print(y_test.values)

# print("\nPredicted values:")
# print(predictions)


# # 7. Calculate accuracy
# accuracy = accuracy_score(y_test, predictions)

# print("\nAccuracy:", accuracy)


# # 8. Test with a new customer

# new_customer = pd.DataFrame({
#     "age": [25, 35, 45, 52, 23, 40, 60, 30, 48, 55],
#     "monthly_bill": [50, 70, 90, 120, 45, 80, 150, 60, 110, 130],
#     "support_calls": [1, 2, 4, 6, 0, 3, 8, 1, 5, 7],
#     "tenure": [24, 36, 12, 5, 30, 20, 3, 40, 8, 4],
# })

# prediction = pipeline.predict(new_customer)

# print("\nNew customer prediction:")

# if prediction[0] == 1:
#     print("Customer is likely to CHURN")
# else:
#     print("Customer is likely to STAY")


# probability = pipeline.predict_proba(new_customer)

# print("\nPrediction probability:")
# print(probability)



import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# -------------------------
# 1. Dataset
# -------------------------

data = {
    "age": [25, 35, 45, 52, 23, 40, 60, 30, 48, 55],
    "monthly_bill": [50, 70, 90, 120, 45, 80, 150, 60, 110, 130],
    "support_calls": [1, 2, 4, 6, 0, 3, 8, 1, 5, 7],
    "tenure": [24, 36, 12, 5, 30, 20, 3, 40, 8, 4],
    "churn": [0, 0, 1, 1, 0, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)


# -------------------------
# 2. Features / Target
# -------------------------

X = df.drop("churn", axis=1)
y = df["churn"]


# -------------------------
# 3. Train / Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# 4. Pipeline
# -------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])


# -------------------------
# 5. Train
# -------------------------

pipeline.fit(X_train, y_train)


# -------------------------
# 6. Predictions
# -------------------------

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]


# -------------------------
# 7. Metrics
# -------------------------

print("Accuracy:",
      accuracy_score(y_test, predictions))

print("Precision:",
      precision_score(y_test, predictions))

print("Recall:",
      recall_score(y_test, predictions))

print("F1:",
      f1_score(y_test, predictions))

print("ROC-AUC:",
      roc_auc_score(y_test, probabilities))


# -------------------------
# 8. Confusion Matrix
# -------------------------

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# -------------------------
# 9. Classification Report
# -------------------------

print(classification_report(y_test, predictions))