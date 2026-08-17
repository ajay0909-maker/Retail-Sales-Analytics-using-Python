import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/ecommerce_sales.csv")
df["order_date"] = pd.to_datetime(df["order_date"])

# Data quality checks
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Feature engineering
df["month"] = df["order_date"].dt.to_period("M").astype(str)
df["profit_margin"] = np.where(df["sales"] != 0, df["profit"] / df["sales"], 0)

# KPIs
total_sales = df["sales"].sum()
total_profit = df["profit"].sum()
total_orders = df["order_id"].nunique()
avg_order_value = total_sales / total_orders
profit_margin = total_profit / total_sales

print(f"\nTotal Sales: ₹{total_sales:,.2f}")
print(f"Total Profit: ₹{total_profit:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Average Order Value: ₹{avg_order_value:,.2f}")
print(f"Profit Margin: {profit_margin:.2%}")

# Analyses
print("\nSales by Category:")
print(df.groupby("category")["sales"].sum().sort_values(ascending=False))

print("\nSales by Region:")
print(df.groupby("region")["sales"].sum().sort_values(ascending=False))

print("\nTop 10 Products:")
print(df.groupby("product")["sales"].sum().sort_values(ascending=False).head(10))

print("\nPayment Modes:")
print(df.groupby("payment_mode")["order_id"].nunique().sort_values(ascending=False))

# Monthly trend
monthly = df.groupby("month")["sales"].sum()
monthly.plot(marker="o", title="Monthly Sales Trend")
plt.ylabel("Sales (₹)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
