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
#print(transaction_revenue.head())
# monthly growth rate

# plot YearMonth revenues
sns.lineplot(x=transaction_revenue['InvoiceYearMonth'],
             y=transaction_revenue['Revenue'].apply(lambda revenue: revenue/1000000), data=transaction_revenue)
plt.title('Monthly Revenue in GBP')
plt.ylabel('Revenue in Millions (M)')

transaction_revenue['MonthlyGrowthRate'] = transaction_revenue['Revenue'].pct_change()
sns.lineplot(x=transaction_revenue['InvoiceYearMonth'],
             y=transaction_revenue['MonthlyGrowthRate'], data=transaction_revenue)
plt.title('Monthly Revenue Growth ')
plt.ylabel('Revenue change (%)')
plt.show()