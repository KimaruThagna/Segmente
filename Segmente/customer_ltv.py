from datetime import date
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

'''
lifetimeValue = Gross Revenue-Total Cost
This will be predicted based on the RFM clustering score
Alternative calculation:
CLTV = Average purchase value(total revenue/number of purchases)*Average purchase frequency(total purchases/number of unique customers)*
customer lifespan(total lifespan/number of customers) 
'''
#read data from csv and redo the data work we done before
transaction_data = pd.read_csv('datasets/OnlineRetail.csv')
transaction_data['InvoiceDate'] = pd.to_datetime(transaction_data['InvoiceDate'])
uk_users = transaction_data.query("Country=='United Kingdom'").reset_index(drop=True)

#3 month dataframe
transaction_data_3m = uk_users[(uk_users.InvoiceDate < date(2011,6,1)) & (uk_users.InvoiceDate >= date(2011,3,1))].reset_index(drop=True)
# 6 months, june to dec
transaction_data_6m = uk_users[(uk_users.InvoiceDate >= date(2011,6,1)) & (uk_users.InvoiceDate < date(2011,12,1))].reset_index(drop=True)
print('3Month Data')
print(transaction_data_3m)
#create user_3m for assigning clustering
user_3m = pd.DataFrame(transaction_data_3m['CustomerID'].unique())
user_3m.columns = ['CustomerID']

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

#calculate recency score
tx_max_purchase = transaction_data_3m.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
user_3m = pd.merge(user_3m, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(user_3m[['Recency']])
user_3m['RecencyCluster'] = kmeans.predict(user_3m[['Recency']])

user_3m = order_cluster('RecencyCluster', 'Recency',user_3m,False)

#calcuate frequency score
tx_frequency = transaction_data_3m.groupby('CustomerID').InvoiceDate.count().reset_index()
tx_frequency.columns = ['CustomerID','Frequency']
user_3m = pd.merge(user_3m, tx_frequency, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(user_3m[['Frequency']])
user_3m['FrequencyCluster'] = kmeans.predict(user_3m[['Frequency']])

user_3m = order_cluster('FrequencyCluster', 'Frequency',user_3m,True)
#calcuate revenue score
transaction_data_3m['Revenue'] = transaction_data_3m['UnitPrice'] * transaction_data_3m['Quantity']
tx_revenue = transaction_data_3m.groupby('CustomerID').Revenue.sum().reset_index()
user_3m = pd.merge(user_3m, tx_revenue, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(user_3m[['Revenue']])
user_3m['RevenueCluster'] = kmeans.predict(user_3m[['Revenue']])
user_3m = order_cluster('RevenueCluster', 'Revenue',user_3m,True)


#overall scoring
user_3m['OverallScore'] = user_3m['RecencyCluster'] + user_3m['FrequencyCluster'] + user_3m['RevenueCluster']
user_3m['Segment'] = user_3m['OverallScore'].apply(lambda value: ('LOW LTV' if value <= 2 else 'MEDIUM LTV') if value < 5 else 'HIGH LTV')
print('3 Month Data with Segmentations')
print(user_3m.head())
# since there is no cost,
#calculate revenue and create a new dataframe for it
transaction_data_6m['Revenue'] = transaction_data_6m['UnitPrice'] * transaction_data_6m['Quantity']
tx_user_6m = transaction_data_6m.groupby('CustomerID')['Revenue'].sum().reset_index()
tx_user_6m.columns = ['CustomerID','m6_Revenue']

# histogram for distribution of LTV

sns.distplot(tx_user_6m.query('m6_Revenue < 10000')['m6_Revenue'], kde=False)
sns.despine(left=True, bottom=True)
plt.title('6 Month LTV (Revenue) Distribution')
plt.ylabel('Number of Customers')
plt.xlabel('Revenue(LTV)')
plt.show()

#merged frames to plot correlation between LTV and futureset
merged_frame = pd.merge(user_3m, tx_user_6m, on='CustomerID', how='left')
merged_frame = merged_frame.fillna(0)

graph_data = merged_frame.query("m6_Revenue < 30000")

#plot correlation between RFM and LTV
sns.scatterplot(x=graph_data['OverallScore'], y=graph_data['m6_Revenue'], hue=graph_data['Segment'], data=graph_data)
sns.despine(left=True, bottom=True)
plt.title('6 Month LTV ')
plt.ylabel('6 Month Revenue')
plt.xlabel('RFM Score')
plt.show()

#remove outliers after studying and finding out the outlier is on the upper side
merged_frame = merged_frame[merged_frame['m6_Revenue']<merged_frame['m6_Revenue'].quantile(0.99)]

#creating 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(merged_frame[['m6_Revenue']])
merged_frame['LTVCluster'] = kmeans.predict(merged_frame[['m6_Revenue']])

#order cluster number based on LTV
merged_frame = order_cluster('LTVCluster', 'm6_Revenue',merged_frame,True)

#creating a new cluster dataframe
clusters = merged_frame.copy()

#see details of the clusters
print('Segmentation based on 6 Months LTV')
print(clusters.groupby('LTVCluster')['m6_Revenue'].describe())

#convert categorical columns to numerical
categorial_classes = pd.get_dummies(merged_frame)

print('LtvCluster column from categorical to numerical')
print(categorial_classes.head())
#calculate and show correlations
corr_matrix = categorial_classes.corr()
corr_matrix['LTVCluster'].sort_values(ascending=False)

print('Correlation Matrix')
print(corr_matrix)
sns.heatmap(corr_matrix)
plt.title('Correlation Matrix')
plt.show()
#create X and y
X = categorial_classes.drop(['LTVCluster','m6_Revenue'],axis=1)
y = categorial_classes['LTVCluster']

#split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)