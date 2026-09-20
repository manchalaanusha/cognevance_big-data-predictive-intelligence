import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("dataset.csv")

# Select features
features = [
    "Age",
    "Purchase_Frequency",
    "Average_Order_Value",
    "Website_Visits",
    "Previous_Purchases",
    "Total_Spending"
]

X = data[features]
y = data["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("===== MODEL RESULTS =====")
print("Model: Logistic Regression")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# Save model predictions
results = X_test.copy()

results["Actual_Churn"] = y_test.values
results["Predicted_Churn"] = y_pred

results.to_csv("model_results.csv", index=False)

print("\nModel results saved to model_results.csv")
print("Predictive modeling completed successfully.")
