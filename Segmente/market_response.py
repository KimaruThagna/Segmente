from datetime import datetime, timedelta,date
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

import sklearn
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# function for ordering cluster numbers for given criteria
def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df, df_new[[cluster_field_name, 'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name], axis=1)
    df_final = df_final.rename(columns={"index": cluster_field_name})
    return df_final


# import the data
df_data = pd.read_csv('datasets/market_response.csv')

# print first 10 rows
print(df_data.head())

'''
Conversion Uplift: Conversion rate of test group - conversion rate of control group
Order Uplift: Conversion uplift * No of converted customer in test group
Revenue Uplift: Order Uplift * Average order value($25)
'''


def calculate_uplift(df):
    # assigning 25$ to the average order value
    avg_order_value = 25

    # calculate conversions for each offer type
    base_conv = df[df.offer == 'No Offer']['conversion'].mean()
    discount_conv = df[df.offer == 'Discount']['conversion'].mean()
    bogo_conv = df[df.offer == 'Buy One Get One']['conversion'].mean()

    # calculate conversion uplift for discount and bogo
    disc_conv_uplift = discount_conv - base_conv
    bogo_conv_uplift = bogo_conv - base_conv

    # calculate order uplift
    discount_order_uplift = disc_conv_uplift * len(df[df.offer == 'Discount']['conversion'])
    bogo_order_uplift = bogo_conv_uplift * len(df[df.offer == 'Buy One Get One']['conversion'])

    # calculate revenue uplift
    discount_rev_uplift = discount_order_uplift * avg_order_value
    bogo_rev_uplift = bogo_order_uplift * avg_order_value
    print('Uplift Report for Dataframe\n')
    print(f'Discount Conversion Uplift: {np.round(disc_conv_uplift * 100, 2)}%')
    print(f'Discount Order Uplift: {np.round(discount_order_uplift, 2)}')
    print(f'Discount Revenue Uplift: ${np.round(discount_rev_uplift, 2)}\n')

    print('-------------- \n')
    print(f'BOGO Conversion Uplift: {np.round(bogo_conv_uplift * 100, 2)}%')
    print(f'BOGO Order Uplift: {np.round(bogo_order_uplift, 2)}')
    print(f'BOGO Revenue Uplift: ${np.round(bogo_rev_uplift, 2)}')
    print('------END-------- \n')

#how different features impact conversion
df_plot = df_data.groupby('recency').conversion.mean().reset_index()
sns.scatterplot(x=df_plot['recency'], y=df_plot['conversion'], data=df_plot)
sns.despine(left=True, bottom=True)
plt.title('Recency vs Conversion')
plt.ylabel('Conversion Rate')
plt.xlabel('Recency (Months)')
plt.show()

sns.scatterplot(x=df_plot['is_referral'], y=df_plot['conversion'], data=df_plot)
sns.despine(left=True, bottom=True)
plt.title('Referral vs Conversion')
plt.ylabel('Conversion Rate')
plt.xlabel('Referral Boolean')
plt.show()

sns.scatterplot(x=df_plot['channel'], y=df_plot['conversion'], data=df_plot)
sns.despine(left=True, bottom=True)
plt.title('Channel vs Conversion')
plt.ylabel('Conversion Rate')
plt.xlabel('Channel')
plt.show()

sns.scatterplot(x=df_plot['offer'], y=df_plot['conversion'], data=df_plot)
sns.despine(left=True, bottom=True)
plt.title('Offer vs Conversion')
plt.ylabel('Conversion Rate')
plt.xlabel('Offer')
plt.show()

kmeans = KMeans(n_clusters=5)
kmeans.fit(df_data[['history']])
df_data['history_cluster'] = kmeans.predict(df_data[['history']])#order the cluster numbers
df_data = order_cluster('history_cluster', 'history',df_data,True)#print how the clusters look like
df_data.groupby('history_cluster').agg({'history':['mean','min','max'], 'conversion':['count', 'mean']})
print('History aggregate data')
print(df_data.head())

df_plot = df_data.groupby('history_cluster').conversion.mean().reset_index()
#higher cluster number means higher history value(value of purchases)
sns.scatterplot(x=df_plot['history_cluster'], y=df_plot['conversion'], data=df_plot)
sns.despine(left=True, bottom=True)
plt.title('History Cluster vs Conversion')
plt.ylabel('Conversion Rate')
plt.xlabel('History')
plt.show()

df_data.groupby(['used_discount','used_bogo','offer']).agg({'conversion':'mean'})
print(df_data.head())

