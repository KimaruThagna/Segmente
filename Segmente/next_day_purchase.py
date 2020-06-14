from datetime import date
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#ml libraries
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

transaction_data = pd.read_csv('datasets/OnlineRetail.csv')
transaction_data['InvoiceDate'] = pd.to_datetime(transaction_data['InvoiceDate'])
uk_users = transaction_data.query("Country=='United Kingdom'").reset_index(drop=True)
# use 6 months of customer purchase data to predict the next purchase date
transaction_6m = uk_users[(uk_users.InvoiceDate < date(2011,9,1)) & (uk_users.InvoiceDate >= date(2011,3,1))].reset_index(drop=True)
transaction_next = uk_users[(uk_users.InvoiceDate >= date(2011,9,1)) & (uk_users.InvoiceDate < date(2011,12,1))].reset_index(drop=True)
# user data
transaction_user = pd.DataFrame(transaction_6m['CustomerID'].unique())
transaction_user.columns = ['CustomerID']
#df with first purchase date after cutoff date
transaction_next_first_purchase = transaction_next.groupby('CustomerID').InvoiceDate.min().reset_index()
transaction_next_first_purchase.columns = ['CustomerID','MinPurchaseDate']
#df with last purchase date before cutoff date
transaction_last_purchase = transaction_6m.groupby('CustomerID').InvoiceDate.max().reset_index()
transaction_last_purchase.columns = ['CustomerID','MaxPurchaseDate']
#merge
transaction_purchase_dates = pd.merge(transaction_last_purchase,transaction_next_first_purchase,on='CustomerID',how='left')
#time delta in days:
transaction_purchase_dates['NextPurchaseDay'] = (transaction_purchase_dates['MinPurchaseDate'] - transaction_purchase_dates['MaxPurchaseDate']).dt.days
#merge with transaction_user 
transaction_user = pd.merge(transaction_user, transaction_purchase_dates[['CustomerID','NextPurchaseDay']],on='CustomerID',how='left')

#print transaction_user
print(transaction_user.head())

#fill NA values with 999
transaction_user = transaction_user.fillna(999)
#get max purchase date for Recency and create a dataframe
transaction_max_purchase = transaction_6m.groupby('CustomerID').InvoiceDate.max().reset_index()
transaction_max_purchase.columns = ['CustomerID','MaxPurchaseDate']

#find the recency in days and add it to transaction_user
transaction_max_purchase['Recency'] = (transaction_max_purchase['MaxPurchaseDate'].max() - transaction_max_purchase['MaxPurchaseDate']).dt.days
transaction_user = pd.merge(transaction_user, transaction_max_purchase[['CustomerID','Recency']], on='CustomerID')
#order cluster method
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


#order recency clusters
transaction_user = order_cluster('RecencyCluster', 'Recency',transaction_user,False)

#print cluster characteristics
print(transaction_user.groupby('RecencyCluster')['Recency'].describe())
#get total purchases for frequency scores
transaction_frequency = transaction_6m.groupby('CustomerID').InvoiceDate.count().reset_index()
transaction_frequency.columns = ['CustomerID','Frequency']

#add frequency column to transaction_user
transaction_user = pd.merge(transaction_user, transaction_frequency, on='CustomerID')
#clustering for frequency
kmeans = KMeans(n_clusters=4)
kmeans.fit(transaction_user[['Frequency']])
transaction_user['FrequencyCluster'] = kmeans.predict(transaction_user[['Frequency']])

#order frequency clusters and show the characteristics
transaction_user = order_cluster('FrequencyCluster', 'Frequency',transaction_user,True)
print(transaction_user.groupby('FrequencyCluster')['Frequency'].describe())

#calculate monetary value, create a dataframe with it
transaction_6m['Revenue'] = transaction_6m['UnitPrice'] * transaction_6m['Quantity']
transaction_revenue = transaction_6m.groupby('CustomerID').Revenue.sum().reset_index()

#add Revenue column to transaction_user
transaction_user = pd.merge(transaction_user, transaction_revenue, on='CustomerID')


#Revenue clusters 
kmeans = KMeans(n_clusters=4)
kmeans.fit(transaction_user[['Revenue']])
transaction_user['RevenueCluster'] = kmeans.predict(transaction_user[['Revenue']])

#ordering clusters and who the characteristics
transaction_user = order_cluster('RevenueCluster', 'Revenue',transaction_user,True)
print(transaction_user.groupby('RevenueCluster')['Revenue'].describe())

#building overall segmentation scores
transaction_user['OverallScore'] = transaction_user['RecencyCluster'] + transaction_user['FrequencyCluster'] + transaction_user['RevenueCluster']
#assign segment names
transaction_user['Segment'] = 'Low-Value'
transaction_user.loc[transaction_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
transaction_user.loc[transaction_user['OverallScore']>4,'Segment'] = 'High-Value'
#plot revenue vs frequency
transaction_graph = transaction_user.query("Revenue < 50000 and Frequency < 2000")

sns.scatterplot(x=transaction_graph['Frequency'], y=transaction_graph['Revenue'], hue=transaction_graph['Segment'], data=transaction_graph)
sns.despine(left=True, bottom=True)
plt.title('User Segmentation Distribution (Frequency vs Revenue)')
plt.ylabel('Revenue')
plt.xlabel('Frequency of Purchase')
plt.show()

transaction_day_order = transaction_6m[['CustomerID','InvoiceDate']]
#convert Invoice Datetime to day
transaction_day_order['InvoiceDay'] = transaction_6m['InvoiceDate'].dt.date
transaction_day_order = transaction_day_order.sort_values(['CustomerID','InvoiceDate'])
#drop duplicates
transaction_day_order = transaction_day_order.drop_duplicates(subset=['CustomerID','InvoiceDay'],keep='first')
# use shifting to create columns of the last 3 purchases
#shifting last 3 purchase dates
transaction_day_order['PrevInvoiceDate'] = transaction_day_order.groupby('CustomerID')['InvoiceDay'].shift(1)
transaction_day_order['T2InvoiceDate'] = transaction_day_order.groupby('CustomerID')['InvoiceDay'].shift(2)
transaction_day_order['T3InvoiceDate'] = transaction_day_order.groupby('CustomerID')['InvoiceDay'].shift(3)
print(transaction_day_order.head())
# difference in days between the last 3 purchases
transaction_day_order['DayDiff'] = (transaction_day_order['InvoiceDay'] - transaction_day_order['PrevInvoiceDate']).dt.days
transaction_day_order['DayDiff2'] = (transaction_day_order['InvoiceDay'] - transaction_day_order['T2InvoiceDate']).dt.days
transaction_day_order['DayDiff3'] = (transaction_day_order['InvoiceDay'] - transaction_day_order['T3InvoiceDate']).dt.days
print(transaction_day_order.head())

#compute mean and standard deviation
transaction_day_diff = transaction_day_order.groupby('CustomerID').agg({'DayDiff': ['mean','std']}).reset_index()
transaction_day_diff.columns = ['CustomerID', 'DayDiffMean','DayDiffStd']
# keep transactions where customers have >3 purchases
transaction_day_order_last = transaction_day_order.drop_duplicates(subset=['CustomerID'],keep='last')

transaction_day_order_last = transaction_day_order_last.dropna() # eliminate na values
transaction_day_order_last = pd.merge(transaction_day_order_last, transaction_day_diff, on='CustomerID')
transaction_user = pd.merge(transaction_user, transaction_day_order_last[['CustomerID','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']], on='CustomerID')
#create transaction_class as a copy of transaction_user before applying get_dummies
transaction_class = transaction_user.copy()
transaction_class = pd.get_dummies(transaction_class)
print(transaction_class.head())
# decide day range in next purchase date to make things eaiser 0-20 cls 2, 21-40 cls1, >50 cls0
transaction_class['NextPurchaseDayRange'] = 2
transaction_class.loc[transaction_class.NextPurchaseDay>20,'NextPurchaseDayRange'] = 1
transaction_class.loc[transaction_class.NextPurchaseDay>50,'NextPurchaseDayRange'] = 0
# correlation matrix
corr = transaction_class[transaction_class.columns].corr()
plt.figure(figsize = (30,20))
sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f")
plt.show()