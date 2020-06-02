from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
'''
NorthStar metric is the metric that captures the core value of your product
for the retail.csv file, the metric will be monthly revenue
'''
# read in CSV
transaction_data = pd.read_csv('datasets/OnlineRetail.csv', header=0)
#print(transaction_data.head())
# convert string date to date time
transaction_data['InvoiceDate'] = pd.to_datetime(transaction_data['InvoiceDate'])
# create yearMonth field for reporting in the sales style. Can use map() or apply()
transaction_data['InvoiceYearMonth'] = transaction_data['InvoiceDate'].map(lambda date: 100*date.year+date.month)
# generate revenue column. Revenue is quantity * UnitPrice
transaction_data['Revenue'] = transaction_data['UnitPrice'] * transaction_data['Quantity']
# generate revenue dataframe
transaction_revenue = transaction_data.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()
transaction_revenue['InvoiceYearMonth'] = transaction_revenue['InvoiceYearMonth'].astype('category')
#print(transaction_revenue.head())
# monthly growth rate

# plot YearMonth revenues
sns.lineplot(x=transaction_revenue['InvoiceYearMonth'],
             y=transaction_revenue['Revenue'].apply(lambda revenue: revenue/1000000), data=transaction_revenue)
plt.title('Monthly Revenue in GBP')
plt.ylabel('Revenue in Millions (M)')
plt.show()

transaction_revenue['MonthlyGrowthRate'] = transaction_revenue['Revenue'].pct_change()

sns.lineplot(x=transaction_revenue['InvoiceYearMonth'],
             y=transaction_revenue['MonthlyGrowthRate'], data=transaction_revenue)
plt.title('Monthly Revenue Growth ')
plt.ylabel('Revenue change (%)')
plt.show()

# monthly active users
uk_users = transaction_data.query("Country=='United Kingdom'").reset_index(drop=True)
#creating monthly active customers dataframe by counting unique Customer IDs nunique returns number of distinct observations
monthly_active_uk_users = uk_users.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()
print(monthly_active_uk_users.head())

sns.barplot(x=monthly_active_uk_users['InvoiceYearMonth'],
             y=monthly_active_uk_users['CustomerID'], data=monthly_active_uk_users)
plt.title('Monthly Active Users ')
plt.ylabel('Number of Active Users')
plt.show()

# monthly order count
transaction_orders = transaction_data.groupby(['InvoiceYearMonth'])['Quantity'].sum().reset_index()
sns.barplot(x=transaction_orders['InvoiceYearMonth'],
             y=transaction_orders['Quantity'], data=transaction_orders)
plt.title('Monthly Orders')
plt.ylabel('Number of orders made')
plt.show()

# average revenue per month
transaction_avg_revenue = transaction_data.groupby(['InvoiceYearMonth'])['Revenue'].mean().reset_index()
sns.barplot(x=transaction_avg_revenue['InvoiceYearMonth'],
             y=transaction_avg_revenue['Revenue'], data=transaction_avg_revenue)
plt.title('Monthly Orders Average Revenue')
plt.ylabel('Average Revenue')
plt.show()

# investigating New and Existing Customers
#create a dataframe contaning CustomerID and first purchase date(also called Min Purchase date)
transaction_min_purchase = monthly_active_uk_users.groupby('CustomerID').InvoiceDate.min().reset_index()
transaction_min_purchase.columns = ['CustomerID','MinPurchaseDate']
transaction_min_purchase['MinPurchaseYearMonth'] = transaction_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)

#merge first purchase date column to our main dataframe (monthly_active_uk_users)
monthly_active_uk_users = pd.merge(monthly_active_uk_users, transaction_min_purchase, on='CustomerID')

print(transaction_min_purchase.head())
