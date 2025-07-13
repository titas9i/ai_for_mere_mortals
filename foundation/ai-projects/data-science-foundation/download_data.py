"""
Download sample datasets for data science practice
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Create data directory
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)
# Generate sample customer data (simulates real e-commerce dataset)
np.random.seed(42)
n_customers = 10000
customer_data = {
    'customer_id': range(1, n_customers + 1),
    'age': np.random.normal(35, 12, n_customers).astype(int),
    'annual_income': np.random.normal(50000, 15000, n_customers),
    'spending_score': np.random.normal(50, 25, n_customers),
    'gender': np.random.choice(['Male', 'Female'], n_customers),
    'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers),
    'membership_years': np.random.exponential(2, n_customers),
    'last_purchase_amount': np.random.lognormal(4, 1, n_customers)
}
# Add some realistic missing data
missing_indices = np.random.choice(n_customers, size=int(n_customers * 0.05), replace=False)
customer_data['annual_income'][missing_indices] = np.nan
df_customers = pd.DataFrame(customer_data)
df_customers.to_csv(data_dir / "customers.csv", index=False)
# Generate sample sales data
n_sales = 50000
sales_data = {
    'sale_id': range(1, n_sales + 1),
    'customer_id': np.random.choice(range(1, n_customers + 1), n_sales),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_sales),
    'sale_amount': np.random.lognormal(3.5, 1, n_sales),
    'sale_date': pd.date_range('2023-01-01', '2024-06-01', periods=n_sales),
    'discount_applied': np.random.choice([0, 5, 10, 15, 20], n_sales),
    'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash'], n_sales)
}
df_sales = pd.DataFrame(sales_data)
df_sales.to_csv(data_dir / "sales.csv", index=False)
print("✅ Sample datasets created successfully!")
print(f"Customers dataset: {len(df_customers)} records")
print(f"Sales dataset: {len(df_sales)} records")
