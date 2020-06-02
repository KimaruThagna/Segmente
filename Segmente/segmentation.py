from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
sns.set()

'''
Performing segmentation using the Recency Frequency and Monetary Model
This mode generates low level, mid and high level customers based on revenue and engagement
'''
# read in CSV
transaction_data = pd.read_csv('datasets/OnlineRetail.csv', header=0)
# calculate Recency- Find out most recent purchase and see how many days theyve been inactive for

transaction_data['InvoiceDate'] = pd.to_datetime(transaction_data['InvoiceDate'])

#we will be using only UK data
uk_users = transaction_data.query("Country=='United Kingdom'").reset_index(drop=True)
user = pd.DataFrame(transaction_data['CustomerID'].unique())# all customers
user.columns = ['CustomerID']

#get the max purchase date for each customer and create a dataframe with it
max_purchase = uk_users.groupby('CustomerID').InvoiceDate.max().reset_index()
max_purchase.columns = ['CustomerID','MaxPurchaseDate'] # max date is also the last date they bought something

#observation point as the max invoice date in our dataset
max_purchase['Recency'] = (max_purchase['MaxPurchaseDate'].max() - max_purchase['MaxPurchaseDate']).dt.days

#merge this dataframe to our new user dataframe
user = pd.merge(user, max_purchase[['CustomerID','Recency']], on='CustomerID')
print('Recency since last purchase in days')
print(user.head())
sns.distplot(user['Recency'],kde=False, bins=10)
sns.despine(left=True, bottom=True)
plt.title('Recency Distribution Histogram')
plt.ylabel('Frequency')
plt.xlabel('Frequency in Days')
plt.show()

# K means clustering
sse={}
recency = user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(recency)
    recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.show()

kmeans = KMeans(n_clusters=4)# start with 4 clusters
kmeans.fit(user[['Recency']])
user['RecencyCluster'] = kmeans.predict(user[['Recency']])
print('Recency Clusters as assigned by KMeans')
print(user.head())

#function for ordering cluster numbers
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    new = new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    new['index'] = new.index
    final = pd.merge(df,new[[cluster_field_name,'index']], on=cluster_field_name)
    final = final.drop([cluster_field_name],axis=1)
    final = final.rename(columns={"index":cluster_field_name})
    return final

user = order_cluster('RecencyCluster', 'Recency',user,False)
print('Ordered Recency Cluster')
print(user.groupby('RecencyCluster')['Recency'].describe())

means = KMeans(n_clusters=4)
kmeans.fit(user[['Frequency']])
user['FrequencyCluster'] = kmeans.predict(user[['Frequency']])

#order the frequency cluster
user = order_cluster('FrequencyCluster', 'Frequency',user,True)

#see details of each cluster
print('Ordered Frequcency Cluster')

print(user.groupby('FrequencyCluster')['Frequency'].describe())