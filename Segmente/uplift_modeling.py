'''
    Treatment Responders(TR): Customers that will purchase only if they receive an offer
    Treatment Non-Responders(TN): Customer that won’t purchase in any case
    Control Responders(CR): Customers that will purchase without an offer(IF YOU OFFER, YOU WILL BE CANNIBALIZING)
    Control Non-Responders(CN): Customers that will not purchase if they don’t receive an offer

'''
## Uplift formula: Uplift score= P(TR)+P(CN)-P(TN)-P(CR)

from datetime import datetime, timedelta, date
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

import sklearn
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# import the data
df_data = pd.read_csv('datasets/market_response.csv')

# print first 10 rows
print(df_data.head())
# function to order clusters
def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df, df_new[[cluster_field_name, 'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name], axis=1)
    df_final = df_final.rename(columns={"index": cluster_field_name})
    return df_final

def calculate_uplift(df, discount=True):
    # assigning 25$ to the average order value
    avg_order_value = 25
    # calculate conversions for each offer type
    base_conv = df[df.offer == 'No Offer']['conversion'].mean()

    print('Uplift Report for Dataframe\n')
    if discount:
        discount_conv = df[df.offer == 'Discount']['conversion'].mean()
        # calculate conversion uplift for discount and bogo
        disc_conv_uplift = discount_conv - base_conv
        # calculate order uplift
        discount_order_uplift = disc_conv_uplift * len(df[df.offer == 'Discount']['conversion'])
        # calculate revenue uplift
        discount_rev_uplift = discount_order_uplift * avg_order_value
        print(f'User count{len(df[df.offer == "Discount"])}')
        print(f'Discount Conversion Uplift: {np.round(disc_conv_uplift * 100, 2)}%')
        print(f'Discount Order Uplift: {np.round(discount_order_uplift, 2)}')
        print(f'Discount Revenue Uplift: ${np.round(discount_rev_uplift, 2)}\n')
        print(f'Revenue uplift per targeted customer ${(np.round(discount_rev_uplift, 2))/len(df[df.offer == "Discount"])}')
    else:
        bogo_conv = df[df.offer == 'Buy One Get One']['conversion'].mean()
        bogo_conv_uplift = bogo_conv - base_conv
        bogo_order_uplift = bogo_conv_uplift * len(df[df.offer == 'Buy One Get One']['conversion'])
        bogo_rev_uplift = bogo_order_uplift * avg_order_value
        print('-------------- \n')
        print(f'User count{len(df[df.offer == "Buy One Get One"])}')
        print(f'BOGO Conversion Uplift: {np.round(bogo_conv_uplift * 100, 2)}%')
        print(f'BOGO Order Uplift: {np.round(bogo_order_uplift, 2)}')
        print(f'BOGO Revenue Uplift: ${np.round(bogo_rev_uplift, 2)}')
        print(f'Revenue uplift per targeted customer ${(np.round(bogo_rev_uplift, 2)) / len(df[df.offer == "Buy One Get One"])}')
    print('------END-------- \n')

#calculate uplift data
calculate_uplift(df_data)
#campaign group column to differentiate treatement and control group
df_data['campaign_group'] = 'treatment'
df_data.loc[df_data.offer == 'No Offer', 'campaign_group'] = 'control'
# labels for various buckets
df_data['target_class'] = 0 #CN
df_data.loc[(df_data.campaign_group == 'control') & (df_data.conversion > 0),'target_class'] = 1 #CR
df_data.loc[(df_data.campaign_group == 'treatment') & (df_data.conversion == 0),'target_class'] = 2 #TN
df_data.loc[(df_data.campaign_group == 'treatment') & (df_data.conversion > 0),'target_class'] = 3 #TR
'''
    0 -> Control Non-Responders
    1 -> Control Responders
    2 -> Treatment Non-Responders
    3 -> Treatment Responders
'''

#creating the clusters
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_data[['history']])
df_data['history_cluster'] = kmeans.predict(df_data[['history']])#order the clusters
df_data = order_cluster('history_cluster', 'history',df_data,True)#creating a new dataframe as model and dropping columns that defines the label
df_model = df_data.drop(['offer','campaign_group','conversion'],axis=1)#convert categorical columns
df_model = pd.get_dummies(df_model)

# fit model and get probabilities
#create feature set and labels
X = df_model.drop(['target_class'],axis=1)
y = df_model.target_class#splitting train and test groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)#fitting the model and predicting the probabilities
xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
class_probs = xgb_model.predict_proba(X_test)
# calculate uplift score
#probabilities for all customers
overall_proba = xgb_model.predict_proba(df_model.drop(['target_class'],axis=1))#assign probabilities to 4 different columns
df_model['proba_CN'] = overall_proba[:,0]
df_model['proba_CR'] = overall_proba[:,1]
df_model['proba_TN'] = overall_proba[:,2]
df_model['proba_TR'] = overall_proba[:,3]#calculate uplift score for all customers
df_model['uplift_score'] = df_model.eval('proba_CN + proba_TR - proba_TN - proba_CR')#assign it back to main dataframe
df_data['uplift_score'] = df_model['uplift_score']

print('Data with Uplift Scores')
print(df_data.head(10))

# model evaluation


print('TARGET DATA')
print('''
Total Targeted Customer Count: 21307
Discount Conversion Uplift: 7.66%
Discount Order Uplift: 1631.89
Discount Revenue Uplift: $40797.35
Revenue Uplift Per Targeted Customer: $1.91
''')

print('UPLIFT MODEL EVALUATION FOR BOGO OFFERS')
df_data_lift = df_data.copy()
uplift_q_75 = df_data_lift.uplift_score.quantile(0.75)
df_data_lift = df_data_lift[(df_data_lift.offer != 'Discount') & (df_data_lift.uplift_score > uplift_q_75)].reset_index(drop=True)
print('Upper Quantile BOGO')
calculate_uplift(df_data_lift, discount=False)
#below 0.5
df_data_lift = df_data.copy()
uplift_q_5 = df_data_lift.uplift_score.quantile(0.5)
df_data_lift = df_data_lift[(df_data_lift.offer != 'Discount') & (df_data_lift.uplift_score < uplift_q_5)].reset_index(drop=True)#calculate the uplift
print('Lower Quantile BOGO')
calculate_uplift(df_data_lift, discount=False)


#DISCOUNT
print('UPLIFT MODEL EVALUATION FOR DISCOUNT OFFERS')
df_data_lift = df_data.copy()
uplift_q_75 = df_data_lift.uplift_score.quantile(0.75)
df_data_lift = df_data_lift[(df_data_lift.offer != 'Buy One Get One') & (df_data_lift.uplift_score > uplift_q_75)].reset_index(drop=True)#calculate the uplift
print('Upper Quantile Discount')
calculate_uplift(df_data_lift)


df_data_lift = df_data.copy()
uplift_q_5 = df_data_lift.uplift_score.quantile(0.5)
df_data_lift = df_data_lift[(df_data_lift.offer != 'Buy One Get One') & (df_data_lift.uplift_score < uplift_q_5)].reset_index(drop=True)#calculate the uplift
print('Lower Quantile Discount')
calculate_uplift(df_data_lift)
