import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset.csv")

print("===== DATASET INFORMATION =====")
print(data.head())

# Check missing values
print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

# Remove duplicate rows
data = data.drop_duplicates()

# Basic statistics
print("\n===== BASIC STATISTICS =====")
print(data.describe())

# KPI calculations
total_customers = len(data)
average_spending = data["Total_Spending"].mean()
average_order_value = data["Average_Order_Value"].mean()
average_visits = data["Website_Visits"].mean()
churn_rate = data["Churn"].mean() * 100

print("\n===== BUSINESS KPIs =====")
print("Total Customers:", total_customers)
print("Average Spending:", round(average_spending, 2))
print("Average Order Value:", round(average_order_value, 2))
print("Average Website Visits:", round(average_visits, 2))
print("Customer Churn Rate:", round(churn_rate, 2), "%")

# Gender analysis
gender_analysis = data.groupby("Gender")["Total_Spending"].mean()

print("\n===== AVERAGE SPENDING BY GENDER =====")
print(gender_analysis)

# Churn analysis
churn_analysis = data.groupby("Churn")["Total_Spending"].mean()

print("\n===== SPENDING BY CHURN STATUS =====")
print(churn_analysis)

# Customer spending chart
plt.figure(figsize=(10, 5))

plt.bar(
    data["Customer_ID"],
    data["Total_Spending"]
)

plt.xlabel("Customer ID")
plt.ylabel("Total Spending")
plt.title("Customer Total Spending")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("customer_spending.png")
plt.close()

# Website visits vs spending
plt.figure(figsize=(8, 5))

plt.scatter(
    data["Website_Visits"],
    data["Total_Spending"]
)

plt.xlabel("Website Visits")
plt.ylabel("Total Spending")
plt.title("Website Visits vs Total Spending")

plt.tight_layout()

plt.savefig("visits_vs_spending.png")
plt.close()

# Churn distribution
churn_counts = data["Churn"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    churn_counts,
    labels=["Stayed", "Churned"],
    autopct="%1.1f%%"
)

plt.title("Customer Churn Distribution")

plt.savefig("churn_distribution.png")
plt.close()

# Save processed data
data.to_csv("processed_dataset.csv", index=False)

print("\nAnalysis completed successfully.")
