import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import xgboost as xgb

churn_data = pd.read_csv('datasets/telco_data.csv')
print(churn_data.head())
churn_data.loc[churn_data.Churn=='No','Churn'] = 0
churn_data.loc[churn_data.Churn=='Yes','Churn'] = 1

# visualize gender distribution of the churn rate
gender_plot = churn_data.groupby('gender').Churn.mean().reset_index()
sns.barplot(x=gender_plot['gender'], y=gender_plot['Churn'], data=gender_plot)
sns.despine(left=True, bottom=True)
plt.title('Gender Distribution of Churn')
plt.ylabel('Number of Churns')
plt.xlabel('Gender')
plt.show()
print('Churn Rate by Gender')
print(churn_data.groupby('gender').Churn.mean())

is_plot = churn_data.groupby('InternetService').Churn.mean().reset_index()
sns.barplot(x=is_plot['InternetService'], y=is_plot['Churn'], data=is_plot)
sns.despine(left=True, bottom=True)
plt.title('InternetService Distribution of Churn')
plt.ylabel('Number of Churns')
plt.xlabel('InternetService')
plt.show()

contract_plot = churn_data.groupby('Contract').Churn.mean().reset_index()
sns.barplot(x=contract_plot['Contract'], y=contract_plot['Churn'], data=contract_plot)
sns.despine(left=True, bottom=True)
plt.title('Contract Distribution of Churn')
plt.ylabel('Number of Churns')
plt.xlabel('Contract Type')
plt.show()

techsupport_plot = churn_data.groupby('TechSupport').Churn.mean().reset_index()
sns.barplot(x=techsupport_plot['TechSupport'], y=techsupport_plot['Churn'], data=techsupport_plot)
sns.despine(left=True, bottom=True)
plt.title('TechSupport Distribution of Churn')
plt.ylabel('Number of Churns')
plt.xlabel('TechSupport Type')
plt.show()

payment_plot = churn_data.groupby('PaymentMethod').Churn.mean().reset_index()
sns.barplot(x=payment_plot['PaymentMethod'], y=payment_plot['Churn'], data=payment_plot)
sns.despine(left=True, bottom=True)
plt.title('Gender Distribution of Churn')
plt.ylabel('Number of Churns')
plt.xlabel('PaymentMethod')
plt.show()
