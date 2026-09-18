# import pandas as pd
# import yaml
# import joblib
# import json
# import matplotlib.pyplot as plt

# from sklearn.metrics import ConfusionMatrixDisplay

# from sklearn.model_selection import train_test_split

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import (
#     StandardScaler,
#     OneHotEncoder
# )

# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     confusion_matrix,
#     classification_report
# )


# # ============================================================
# # 1. Load configuration
# # ============================================================

# with open("config.yaml", "r") as file:
#     config = yaml.safe_load(file)

# print("Configuration:")
# print(config)


# # ============================================================
# # 2. Load dataset
# # ============================================================

# df = pd.read_csv(config["data"]["path"])

# print("\nFirst 5 rows:")
# print(df.head())

# print("\nDataset information:")
# df.info()


# # ============================================================
# # 3. Separate features and target
# # ============================================================

# X = df.drop("churn", axis=1)
# y = df["churn"]


# # ============================================================
# # 4. Define numerical and categorical features
# # ============================================================

# numeric_features = [
#     "age",
#     "monthly_bill",
#     "tenure",
#     "support_calls"
# ]

# categorical_features = [
#     "contract",
#     "payment",
#     "internet_service"
# ]


# # ============================================================
# # 5. Train/Test Split
# # ============================================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=config["training"]["test_size"],
#     random_state=config["training"]["random_state"],
#     stratify=y
# )

# print("\nTraining records:", len(X_train))
# print("Testing records:", len(X_test))


# # ============================================================
# # 6. Create preprocessing pipeline
# # ============================================================

# preprocessor = ColumnTransformer(
#     transformers=[
#         (
#             "num",
#             StandardScaler(),
#             numeric_features
#         ),
#         (
#             "cat",
#             OneHotEncoder(handle_unknown="ignore"),
#             categorical_features
#         )
#     ]
# )


# # ============================================================
# # 7. Create ML Pipeline
# # ============================================================

# pipeline = Pipeline([
#     (
#         "preprocessor",
#         preprocessor
#     ),
#     (
#         "model",
#         LogisticRegression(
#             max_iter=config["model"]["max_iter"]
#         )
#     )
# ])


# # ============================================================
# # 8. Train the model
# # ============================================================

# pipeline.fit(X_train, y_train)

# print("\nModel training completed!")


# # ============================================================
# # 9. Make predictions
# # ============================================================

# predictions = pipeline.predict(X_test)

# probabilities = pipeline.predict_proba(X_test)[:, 1]


# # ============================================================
# # 10. Calculate metrics
# # ============================================================

# accuracy = accuracy_score(
#     y_test,
#     predictions
# )

# precision = precision_score(
#     y_test,
#     predictions
# )

# recall = recall_score(
#     y_test,
#     predictions
# )

# f1 = f1_score(
#     y_test,
#     predictions
# )

# roc_auc = roc_auc_score(
#     y_test,
#     probabilities
# )


# # ============================================================
# # 11. Print metrics
# # ============================================================

# print("\n===== MODEL METRICS =====")

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1:", f1)
# print("ROC-AUC:", roc_auc)


# # ============================================================
# # 12. Confusion Matrix
# # ============================================================

# print("\n===== CONFUSION MATRIX =====")

# cm = confusion_matrix(
#     y_test,
#     predictions
# )

# print(cm)


# # ============================================================
# # 13. Classification Report
# # ============================================================

# print("\n===== CLASSIFICATION REPORT =====")

# print(
#     classification_report(
#         y_test,
#         predictions
#     )
# )


# # ============================================================
# # 14. Save trained model
# # ============================================================

# model_path = config["output"]["model_path"]

# joblib.dump(
#     pipeline,
#     model_path
# )

# print("\nModel saved successfully!")
# print("Model path:", model_path)


# # ============================================================
# # 15. Save metrics
# # ============================================================

# metrics = {
#     "accuracy": accuracy,
#     "precision": precision,
#     "recall": recall,
#     "f1": f1,
#     "roc_auc": roc_auc
# }

# metrics_path = config["output"]["metrics_path"]

# with open(metrics_path, "w") as file:
#     json.dump(
#         metrics,
#         file,
#         indent=4
#     )

# print("Metrics saved successfully!")
# print("Metrics path:", metrics_path)


# ConfusionMatrixDisplay.from_predictions(
#     y_test,
#     predictions
# )

# plt.title("Churn Prediction Confusion Matrix")

# plt.savefig(
#     "outputs/confusion_matrix.png",
#     bbox_inches="tight"
# )

# plt.close()


# error_df = X_test.copy()

# error_df["actual"] = y_test.values

# error_df["predicted"] = predictions

# error_df["probability"] = probabilities

# error_df["correct"] = (
#     error_df["actual"] ==
#     error_df["predicted"]
# )

# misclassified = error_df[
#     error_df["actual"] !=
#     error_df["predicted"]
# ]

# print(misclassified)


# misclassified.to_csv(
#     "outputs/misclassified.csv",
#     index=False
# )

# misclassified["confidence"] = (
#     abs(
#         misclassified["probability"] - 0.5
#     )
# )

# misclassified = misclassified.sort_values(
#     "confidence",
#     ascending=False
# )


import os
import json
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)


# -----------------------------
# 1. Load configuration
# -----------------------------

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


data_path = config["data"]["path"]

model_path = config["output"]["model_path"]
metrics_path = config["output"]["metrics_path"]
confusion_path = config["output"]["confusion_matrix_path"]
errors_path = config["output"]["errors_path"]

test_size = config["training"]["test_size"]
random_state = config["training"]["random_state"]

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# -----------------------------
# 2. Load data
# -----------------------------

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print(df.head())


# -----------------------------
# 3. Select features
# -----------------------------

features = [
    "age",
    "monthly_bill",
    "tenure",
    "support_calls",
    "contract",
    "payment",
    "internet_service"
]

target = "churn"

X = df[features]
y = df[target]


# -----------------------------
# 4. Train/Test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)


# -----------------------------
# 5. Feature types
# -----------------------------

numeric_features = [
    "age",
    "monthly_bill",
    "tenure",
    "support_calls"
]

categorical_features = [
    "contract",
    "payment",
    "internet_service"
]


# -----------------------------
# 6. Preprocessing
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# -----------------------------
# 7. Create pipeline
# -----------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=config["model"]["max_iter"]
            )
        )
    ]
)


# -----------------------------
# 8. Train
# -----------------------------

print("Training model...")

pipeline.fit(X_train, y_train)

print("Training completed.")


# -----------------------------
# 9. Predictions
# -----------------------------

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]


# -----------------------------
# 10. Metrics
# -----------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc
}


print("\nModel Metrics")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


# -----------------------------
# 11. Save metrics
# -----------------------------

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)


# -----------------------------
# 12. Confusion Matrix
# -----------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions
)

plt.title("Churn Confusion Matrix")
plt.tight_layout()

plt.savefig(confusion_path)

plt.close()


# -----------------------------
# 13. Error Analysis
# -----------------------------

error_df = X_test.copy()

error_df["actual"] = y_test.values

error_df["predicted"] = predictions

error_df["probability"] = probabilities

misclassified = error_df[
    error_df["actual"] != error_df["predicted"]
].copy()

misclassified["confidence"] = (
    abs(misclassified["probability"] - 0.5)
)

misclassified = misclassified.sort_values(
    "confidence",
    ascending=False
)

misclassified.head(20).to_csv(
    errors_path,
    index=False
)


print(
    f"\nMisclassified records: {len(misclassified)}"
)


# -----------------------------
# 14. Save model
# -----------------------------

joblib.dump(
    pipeline,
    model_path
)

print("\nModel saved to:", model_path)

print("Training pipeline completed successfully.")