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
sns.lineplot(x=transaction_revenue['InvoiceYearMonth'].astype(str),
             y=transaction_revenue['Revenue'].apply(lambda revenue: revenue/1000000), data=transaction_revenue)
plt.title('Monthly Revenue in GBP')
plt.ylabel('Revenue in Millions (M)')
plt.show()

transaction_revenue['MonthlyGrowthRate'] = transaction_revenue['Revenue'].pct_change()

sns.lineplot(x=transaction_revenue['InvoiceYearMonth'].astype(str),
             y=transaction_revenue['MonthlyGrowthRate'], data=transaction_revenue)
plt.title('Monthly Revenue Growth ')
plt.ylabel('Revenue change (%)')
plt.show()

# monthly active users
uk_users = transaction_data.query("Country=='United Kingdom'").reset_index(drop=True)
#creating monthly active customers dataframe by counting unique Customer IDs nunique returns number of distinct observations
monthly_active_uk_users = uk_users.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()
print('Monthly Active Users')
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
plt.ylabel('Average Revenue per Order')
plt.show()

# investigating New and Existing Customers
#create a dataframe contaning CustomerID and first purchase date(also called Min Purchase date)
transaction_min_purchase = uk_users.groupby('CustomerID').InvoiceDate.min().reset_index()
transaction_min_purchase.columns = ['CustomerID','MinPurchaseDate']
transaction_min_purchase['MinPurchaseYearMonth'] = transaction_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)

#merge first purchase date column to our main dataframe (monthly_active_uk_users)
monthly_active_uk_users = pd.merge(uk_users, transaction_min_purchase, on='CustomerID')
print('First Purchase Date per Customer')
print(transaction_min_purchase.head())

#create a column called User Type and assign Existing if User's First Purchase Year Month before the selected Invoice Year Month
monthly_active_uk_users['UserType'] = 'New'
monthly_active_uk_users.loc[monthly_active_uk_users['InvoiceYearMonth']>monthly_active_uk_users['MinPurchaseYearMonth'],'UserType'] = 'Existing'

#Revenue per month for each user type
transaction_user_type_revenue = monthly_active_uk_users.groupby(['InvoiceYearMonth','UserType'])['Revenue'].sum().reset_index()

#filtering the dates and plot the result
transaction_user_type_revenue = transaction_user_type_revenue.query("InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112")
print('User Type Revenue')
print(transaction_user_type_revenue.head())

sns.lineplot(x=transaction_user_type_revenue['InvoiceYearMonth'].astype(str),
             y=transaction_user_type_revenue['Revenue'], hue=transaction_user_type_revenue['UserType'],data=transaction_user_type_revenue)
plt.title('Monthly Revenues Based on User Type')
plt.ylabel('Average Monthly Revenue')
plt.show()

# Determining Customer Ratio
#create a dataframe that shows new user ratio(new:existing) - we also need to drop NA values (first month new user ratio is 0)
user_ratio = monthly_active_uk_users.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()/monthly_active_uk_users.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()
user_ratio = user_ratio.reset_index()
user_ratio = user_ratio.dropna()

#print the dafaframe
print('User Ratio, New/Existing')
print(user_ratio.head())

sns.barplot(x=user_ratio['InvoiceYearMonth'],
             y=user_ratio['CustomerID'],data=user_ratio)
plt.title('New vs Existing Customers Monthly Ratios')
plt.ylabel('New/Existing')
plt.show()

# Determining Monthly Retention Rate. How good your business is at making customers come back

#identify which users are active by looking at their revenue per month
user_purchase = uk_users.groupby(['CustomerID','InvoiceYearMonth'])['Revenue'].sum().reset_index()
print("User Purchases Per Month")
print(user_purchase.head())
# build a customer retention table
user_retention = pd.crosstab(user_purchase['CustomerID'], user_purchase['InvoiceYearMonth']).reset_index()
print('User Retention')
print(user_retention.head())
months = user_retention.columns[2:]
retention_array = []
for i in range(len(months) - 1):
    retention_data = {}
    selected_month = months[i + 1]
    prev_month = months[i]
    retention_data['InvoiceYearMonth'] = int(selected_month)
    retention_data['TotalUserCount'] = user_retention[selected_month].sum()
    retention_data['RetainedUserCount'] = \
    user_retention[(user_retention[selected_month] > 0) & (user_retention[prev_month] > 0)][selected_month].sum()
    retention_array.append(retention_data)

# convert the array to dataframe and calculate Retention Rate
user_retention = pd.DataFrame(retention_array)
user_retention['RetentionRate'] = user_retention['RetainedUserCount'] / user_retention['TotalUserCount']
print('Retention Rate')
print(user_retention)

sns.barplot(x=user_retention['InvoiceYearMonth'],
             y=user_retention['RetentionRate'],data=user_retention)
plt.title('Retention Rate')
plt.ylabel('Retention Rate')
plt.show()


# Cohort based Retention Rate
new_column_names = ['m_' + str(column) for column in user_retention.columns]
user_retention.columns = new_column_names

# create the array of Retained users for each cohort monthly
retention_array = []
for i in range(len(months)):
    retention_data = {}
    selected_month = months[i]
    prev_months = months[:i]
    next_months = months[i + 1:]
    for prev_month in prev_months:
        retention_data[prev_month] = np.nan # cannot count retentaion rate for months before the current when considering the current

    total_user_count = retention_data['TotalUserCount'] = user_retention['m_' + str(selected_month)].sum()
    retention_data[selected_month] = 1

    query = "{} > 0".format('m_' + str(selected_month))

    for next_month in next_months:
        query = query + " and {} > 0".format(str('m_' + str(next_month)))
        retention_data[next_month] = np.round(
            user_retention.query(query)['m_' + str(next_month)].sum() / total_user_count, 2)
    retention_array.append(retention_data)

user_retention = pd.DataFrame(retention_array)
user_retention.index = months

# showing new cohort based retention table
print('Cohort Based Retention Table')
print(user_retention.head())
