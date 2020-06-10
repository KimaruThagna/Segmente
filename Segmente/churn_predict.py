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