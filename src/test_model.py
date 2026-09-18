import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/churn_model.joblib")

print("Model loaded successfully!")

# New customer data
customer = pd.DataFrame([
    {
        "age": 45,
        "monthly_bill": 90,
        "tenure": 12,
        "contract": "Month-to-month",
        "payment": "Electronic",
        "support_calls": 4,
        "internet_service": "Fiber"
    }
])

# Make prediction
prediction = model.predict(customer)

# Get churn probability
probability = model.predict_proba(customer)[0][1]

print("\nPrediction:", prediction[0])
print("Churn probability:", probability)

if prediction[0] == 1:
    print("Customer is likely to CHURN")
else:
    print("Customer is likely to STAY")