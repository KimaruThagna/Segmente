import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import xgboost as xgb

churn_data = pd.read_csv('datasets/telco_data.csv')
churn_data.loc[churn_data.Churn=='No','Churn'] = 0
churn_data.loc[churn_data.Churn=='Yes','Churn'] = 1

print(churn_data.head())
print(churn_data.info())
# visualize gender distribution of the churn rate
gender_plot = churn_data.groupby('gender').Churn.mean().reset_index()
sns.barplot(x=gender_plot['gender'], y=gender_plot['Churn'], data=gender_plot)
sns.despine(left=True, bottom=True)
plt.title('Gender Distribution of Churn')
plt.ylabel('Churn Rate')
plt.xlabel('Gender')
plt.show()
print('Churn Rate by Gender')
print(churn_data.groupby('gender').Churn.mean())

is_plot = churn_data.groupby('InternetService').Churn.mean().reset_index()
sns.barplot(x=is_plot['InternetService'], y=is_plot['Churn'], data=is_plot)
sns.despine(left=True, bottom=True)
plt.title('InternetService Distribution of Churn')
plt.ylabel('Churn Rate')
plt.xlabel('InternetService')
plt.show()

contract_plot = churn_data.groupby('Contract').Churn.mean().reset_index()
sns.barplot(x=contract_plot['Contract'], y=contract_plot['Churn'], data=contract_plot)
sns.despine(left=True, bottom=True)
plt.title('Contract Distribution of Churn')
plt.ylabel('Churn Rate')
plt.xlabel('Contract Type')
plt.show()

techsupport_plot = churn_data.groupby('TechSupport').Churn.mean().reset_index()
sns.barplot(x=techsupport_plot['TechSupport'], y=techsupport_plot['Churn'], data=techsupport_plot)
sns.despine(left=True, bottom=True)
plt.title('TechSupport Distribution of Churn')
plt.ylabel('Churn Rate')
plt.xlabel('TechSupport Type')
plt.show()

payment_plot = churn_data.groupby('PaymentMethod').Churn.mean().reset_index()
sns.barplot(x=payment_plot['PaymentMethod'], y=payment_plot['Churn'], data=payment_plot)
sns.despine(left=True, bottom=True)
plt.title('Payment Method Distribution of Churn')
plt.ylabel('Churn Rate')
plt.xlabel('PaymentMethod')
plt.show()

# churn rate vs Numerical variables

tenure_plot = churn_data.groupby('tenure').Churn.mean().reset_index()
sns.scatterplot(x=tenure_plot['tenure'], y=tenure_plot['Churn'], data=tenure_plot)
sns.despine(left=True, bottom=True)
plt.title('tenure Distribution of Churn')
plt.ylabel('Churn Rate')
plt.xlabel('Tenure')
plt.show()


charge_plot = churn_data.groupby('MonthlyCharges').Churn.mean().reset_index()
sns.scatterplot(x=charge_plot['MonthlyCharges'], y=charge_plot['Churn'], data=charge_plot)
sns.despine(left=True, bottom=True)
plt.title('MonthlyCharges Distribution of Churn')
plt.ylabel('Churn Rate')
plt.xlabel('MonthlyCharges')
plt.show()


TotalCharges_plot = churn_data.groupby('TotalCharges').Churn.mean().reset_index()
sns.scatterplot(x=TotalCharges_plot['TotalCharges'], y=TotalCharges_plot['Churn'], data=TotalCharges_plot)
sns.despine(left=True, bottom=True)
plt.title('tenure Distribution of Churn')
plt.ylabel('Churn Rate')
plt.xlabel('TotalCharges')
plt.show()

#feature engineering
#create Clusters for Tenure, monthly Charge and total charge


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

#tenure clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(churn_data[['tenure']])
churn_data['TenureCluster'] = kmeans.predict(churn_data[['tenure']])
print(churn_data.groupby('TenureCluster').tenure.describe())